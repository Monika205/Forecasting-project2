import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = FastAPI(title="Monthly Demand Forecasting API")


class MonthDemand(BaseModel):
    month: str
    demand: float = Field(ge=0)


class ForecastRequest(BaseModel):
    data: List[MonthDemand]
    months: int = Field(default=3, ge=1, le=24)


@app.get("/")
def home():
    return {
        "message": "Monthly Demand Forecasting API is running",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        if len(request.data) < 6:
            raise HTTPException(status_code=400, detail="At least 6 months of data required.")

        df = pd.DataFrame([row.model_dump() for row in request.data])
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

        if df["month"].isna().any() or df["demand"].isna().any():
            raise HTTPException(status_code=400, detail="Each row needs a valid YYYY-MM month and numeric demand.")
        if (df["demand"] < 0).any():
            raise HTTPException(status_code=400, detail="Demand cannot be negative.")

        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
        df = df.sort_values("month")
        if df["month"].duplicated().any():
            raise HTTPException(status_code=400, detail="Duplicate months found. Keep one row per month.")

        expected = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
        if len(expected) != len(df) or not df["month"].reset_index(drop=True).equals(pd.Series(expected, name="month")):
            raise HTTPException(status_code=400, detail="Months must be consecutive. Add any missing months.")

        monthly = pd.Series(df["demand"].to_numpy(float), index=expected, name="Demand")
        monthly.index.name = "Month"

        test_size = min(3, max(1, len(monthly) // 5))
        train, test = monthly.iloc[:-test_size], monthly.iloc[-test_size:]
        scores = []

        naive_pred = np.repeat(float(train.iloc[-1]), len(test))
        scores.append({
            "model": "Naive baseline",
            "MAE": round(float(mean_absolute_error(test, naive_pred)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(test, naive_pred))), 2),
        })

        candidates = {}
        try:
            candidates["Holt Trend"] = lambda series, steps: Holt(
                series, initialization_method="estimated"
            ).fit().forecast(steps)
        except Exception:
            pass

        if len(train) >= 6:
            try:
                candidates["Holt Trend"] = lambda series, steps: Holt(
                    series, initialization_method="estimated"
                ).fit().forecast(steps)
                pred = np.asarray(candidates["Holt Trend"](train, len(test)), dtype=float)
                scores.append({
                    "model": "Holt Trend",
                    "MAE": round(float(mean_absolute_error(test, pred)), 2),
                    "RMSE": round(float(np.sqrt(mean_squared_error(test, pred))), 2),
                })
            except Exception:
                pass

        if len(train) >= 24:
