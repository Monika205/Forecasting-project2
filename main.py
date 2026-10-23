import pandas as pd
import numpy as np
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException


from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing, Holt, ExponentialSmoothing
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = FastAPI(title="Demand Forecast API")


@app.get("/")
def home():
    return {
        "message": "Monthly Demand Forecasting API is running",
        "health": "/health",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast")
async def forecast(file: UploadFile = File(...), months: int = 3):
    if months not in [3, 6, 12]:
        raise HTTPException(400, "Choose 3, 6, or 12 forecast months")

    try:
        content = await file.read()
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(content))
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(400, "Upload CSV or Excel")

        cols = {str(c).strip().lower(): c for c in df.columns}
        date_col = next((cols[c] for c in ["orderdate", "order date", "date"] if c in cols), None)
        qty_col = next((cols[c] for c in ["quantity", "qty", "demand", "sales"] if c in cols), None)

        if not date_col or not qty_col:
            raise HTTPException(400, "Date and Quantity columns required")

        df = df[[date_col, qty_col]].copy()
        df.columns = ["date", "quantity"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df = df.dropna()

        monthly = df.set_index("date")["quantity"].resample("MS").sum()
        monthly = monthly.asfreq("MS", fill_value=0).astype(float)

        if len(monthly) < 3:
            raise HTTPException(400, "At least 3 months of data required")

        test_size = min(12, max(1, len(monthly) // 4))
        train, test = monthly.iloc[:-test_size], monthly.iloc[-test_size:]
        scores = []

        models = {
            "Simple Exponential Smoothing": lambda: SimpleExpSmoothing(
                train, initialization_method="estimated"
            ).fit().forecast(len(test)),
            "Holt Trend": lambda: Holt(
                train, initialization_method="estimated"
            ).fit().forecast(len(test))
        }

        if len(train) >= 24:
            models["Holt-Winters Seasonal"] = lambda: ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=12,
                initialization_method="estimated"
            ).fit().forecast(len(test))

        for name, model_fn in models.items():
            try:
                pred = np.asarray(model_fn())
                scores.append({
                    "model": name,
                    "MAE": round(float(mean_absolute_error(test, pred)), 2),
                    "RMSE": round(float(np.sqrt(mean_squared_error(test, pred))), 2)
                })
            except Exception:
                pass

        scores.sort(key=lambda x: x["MAE"])
        best = scores[0]["model"] if scores else "Simple Exponential Smoothing"

        if best == "Holt-Winters Seasonal" and len(monthly) >= 24:
            fitted = ExponentialSmoothing(
                monthly, trend="add", seasonal="add", seasonal_periods=12,
                initialization_method="estimated"
            ).fit()
        elif best == "Holt Trend":
            fitted = Holt(monthly, initialization_method="estimated").fit()
        else:
            fitted = SimpleExpSmoothing(
                monthly, initialization_method="estimated"
            ).fit()

        predictions = fitted.forecast(months).clip(lower=0)

        return {
            "summary": {
                "months": len(monthly),
                "total_quantity": float(monthly.sum()),
                "average_demand": round(float(monthly.mean()), 2)
            },
            "best_model": best,
            "model_scores": scores,
            "history": [
                {"month": d.strftime("%Y-%m"), "quantity": float(v)}
                for d, v in monthly.items()
            ],
            "forecast": [
                {"month": d.strftime("%Y-%m"), "quantity": int(round(v))}
                for d, v in predictions.items()
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))
