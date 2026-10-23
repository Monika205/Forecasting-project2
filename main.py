import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt

warnings.filterwarnings("ignore")

app = FastAPI(title="Monthly Demand Forecasting API")


class MonthDemand(BaseModel):
    month: str
    demand: float = Field(ge=0)


class ForecastRequest(BaseModel):
    data: List[MonthDemand]
    months: int = Field(default=3, ge=1, le=24)
    # Optional: use only the most recent N months. If omitted, use all submitted history.
    history_months: Optional[int] = Field(default=None, ge=6, le=60)


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


def fit_predict(model_name: str, values, steps: int) -> np.ndarray:
    """Fit the named forecasting model and return non-negative forecasts."""
    y = pd.Series(np.asarray(values, dtype=float))

    if model_name == "Naive baseline":
        forecast_values = np.repeat(float(y.iloc[-1]), steps)
    elif model_name == "Mean baseline":
        forecast_values = np.repeat(float(y.mean()), steps)
    elif model_name == "Drift":
        if len(y) < 2:
            forecast_values = np.repeat(float(y.iloc[-1]), steps)
        else:
            slope = (float(y.iloc[-1]) - float(y.iloc[0])) / (len(y) - 1)
            forecast_values = float(y.iloc[-1]) + slope * np.arange(1, steps + 1)
    elif model_name == "Holt damped trend":
        fitted = Holt(
            y,
            damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast_values = fitted.forecast(steps)
    elif model_name == "Holt-Winters seasonal":
        if len(y) < 24:
            raise ValueError("Holt-Winters needs at least 24 months of history.")
        fitted = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast_values = fitted.forecast(steps)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return np.maximum(np.asarray(forecast_values, dtype=float), 0.0)


@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        if len(request.data) < 6:
            raise HTTPException(
                status_code=400,
                detail="At least 6 months of historical data are required.",
            )

        # .dict() supports both Pydantic v1 and v2.
        df = pd.DataFrame([row.dict() for row in request.data])
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

        if df["month"].isna().any() or df["demand"].isna().any():
            raise HTTPException(
