
import pandas as pd
import numpy as np
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing, Holt, ExponentialSmoothing
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = FastAPI(title="Monthly Demand Forecast API")


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


def prepare_data(df):
    # Match common date and demand column names
    cols = {str(c).strip().lower(): c for c in df.columns}

    date_col = next(
        (cols[c] for c in [
            "orderdate", "order date", "date", "month",
            "month (yyyy-mm-)", "month (yyyy-mm)"
        ] if c in cols),
        None
    )

    qty_col = next(
        (cols[c] for c in [
            "quantity", "qty", "demand", "sales"
        ] if c in cols),
        None
    )

    if date_col is None or qty_col is None:
        raise HTTPException(
            status_code=400,
            detail="Date/Month and Quantity/Demand columns are required."
        )

    df = df[[date_col, qty_col]].copy()
    df.columns = ["date", "quantity"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna()

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid date and demand data found."
        )

    if (df["quantity"] < 0).any():
        raise HTTPException(
            status_code=400,
            detail="Demand quantities cannot be negative."
        )

    monthly = (
        df.set_index("date")["quantity"]
        .resample("MS")
        .sum()
    )

    monthly = monthly.asfreq("MS", fill_value=0).astype(float)

    if len(monthly) < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 months of data are required."
        )

    return monthly


@app.post("/forecast")
async def forecast(
    request: Request,
    file: UploadFile = File(None),
    months: int = 3
):
    if not 1 <= months <= 24:
        raise HTTPException(
            status_code=400,
            detail="Forecast months must be between 1 and 24."
        )

    try:
        content_type = request.headers.get("content-type", "").lower()

        # Option 1: Existing CSV/Excel upload
        if file is not None:
            content = await file.read()
            filename = (file.filename or "").lower()

            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Please upload a CSV or Excel file."
                )

        # Option 2: Manually entered frontend data sent as JSON
        elif "application/json" in content_type:
            payload = await request.json()

            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict):
                rows = (
                    payload.get("data")
                    or payload.get("history")
                    or payload.get("monthly_data")
                )
                if rows is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "JSON must contain a data, history, "
                            "or monthly_data list."
                        )
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON data format."
                )

            df = pd.DataFrame(rows)

        else:
            raise HTTPException(
                status_code=400,
                detail="Send a CSV/Excel file or JSON demand data."
            )

        monthly = prepare_data(df)

        # Evaluate forecasting models using recent observations
        test_size = min(12, max(1, len(monthly) // 4))
        train = monthly.iloc[:-test_size]
        test = monthly.iloc[-test_size:]
        scores = []

        models = {
            "Simple Exponential Smoothing": lambda: (
                SimpleExpSmoothing(
                    train, initialization_method="estimated"
                ).fit().forecast(len(test))
            ),
            "Holt Trend": lambda: (
                Holt(
                    train, initialization_method="estimated"
                ).fit().forecast(len(test))
            )
        }

        if len(train) >= 24:
            models["Holt-Winters Seasonal"] = lambda: (
                ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=12,
                    initialization_method="estimated"
                ).fit().forecast(len(test))
            )

        for name, model_fn in models.items():
            try:
                pred = np.asarray(model_fn())
                scores.append({
                    "model": name,
                    "MAE": round(
                        float(mean_absolute_error(test, pred)), 2
                    ),
                    "RMSE": round(
                        float(np.sqrt(mean_squared_error(test, pred))), 2
                    )
                })
            except Exception:
                continue

        scores.sort(key=lambda x: x["MAE"])
        best = (
            scores[0]["model"]
            if scores
            else "Simple Exponential Smoothing"
        )

        # Fit the selected model on all available history
        if best == "Holt-Winters Seasonal" and len(monthly) >= 24:
            fitted = ExponentialSmoothing(
                monthly,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated"
            ).fit()
        elif best == "Holt Trend":
            fitted = Holt(
                monthly, initialization_method="estimated"
            ).fit()
        else:
            fitted = SimpleExpSmoothing(
                monthly, initialization_method="estimated"
            ).fit()

        predictions = fitted.forecast(months).clip(lower=0)

        return {
            "summary": {
                "months": int(len(monthly)),
                "total_quantity": float(monthly.sum()),
                "average_demand": round(float(monthly.mean()), 2)
            },
            "best_model": best,
            "model_scores": scores,
            "history": [
                {
                    "month": d.strftime("%Y-%m"),
                    "quantity": float(v)
                }
                for d, v in monthly.items()
            ],
            "forecast": [
                {
                    "month": d.strftime("%Y-%m"),
                    "quantity": int(round(v))
                }
                for d, v in predictions.items()
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process forecast data: {str(e)}"
        )
