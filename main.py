import warnings
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing

warnings.filterwarnings("ignore")
app = FastAPI(title="Monthly Demand Forecasting API")


class MonthDemand(BaseModel):
    month: str
    demand: float = Field(ge=0)


class ForecastRequest(BaseModel):
    data: List[MonthDemand]
    months: int = Field(default=3, ge=3, le=5)
    history_months: int = Field(default=12, ge=6, le=24)


@app.get("/")
def home():
    return {"message": "Monthly Demand Forecasting API is running", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


def fit_predict(name, series, steps):
    """Fit one candidate model and return non-negative forecasts."""
    y = pd.Series(np.asarray(series, dtype=float))
    if name == "Naive baseline":
        values = np.repeat(float(y.iloc[-1]), steps)
    elif name == "Mean baseline":
        values = np.repeat(float(y.mean()), steps)
    elif name == "Drift":
        if len(y) < 2:
            values = np.repeat(float(y.iloc[-1]), steps)
        else:
            slope = (float(y.iloc[-1]) - float(y.iloc[0])) / (len(y) - 1)
            values = float(y.iloc[-1]) + slope * np.arange(1, steps + 1)
    elif name == "Holt damped trend":
        values = Holt(y, damped_trend=True, initialization_method="estimated").fit(optimized=True).forecast(steps)
    elif name == "Holt-Winters seasonal":
        if len(y) < 24:
            raise ValueError("Need at least 24 observations for yearly seasonality.")
        values = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=12, initialization_method="estimated"
        ).fit(optimized=True).forecast(steps)
    else:
        raise ValueError("Unknown model")
    return np.maximum(np.asarray(values, dtype=float), 0.0)


@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        if len(request.data) < 6:
            raise HTTPException(status_code=400, detail="Enter at least 6 months of historical data.")

        df = pd.DataFrame([row.model_dump() for row in request.data])
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
        if df["month"].isna().any() or df["demand"].isna().any():
            raise HTTPException(status_code=400, detail="Use valid YYYY-MM months and numeric demand.")
        if (df["demand"] < 0).any():
            raise HTTPException(status_code=400, detail="Demand cannot be negative.")

        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
        df = df.sort_values("month").reset_index(drop=True)
        if df["month"].duplicated().any():
            raise HTTPException(status_code=400, detail="Duplicate months found. Keep one row per month.")

        expected = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
        if len(expected) != len(df) or not np.array_equal(df["month"].to_numpy(), expected.to_numpy()):
            raise HTTPException(status_code=400, detail="Months must be consecutive. Add missing months.")

        all_series = pd.Series(df["demand"].to_numpy(dtype=float), index=expected, name="Demand")
        # Use only the most recent user-selected history window for fitting.
        window = min(int(request.history_months), len(all_series))
        monthly = all_series.iloc[-window:].copy()
        monthly.index.name = "Month"

        # Rolling-origin backtest: each candidate predicts the next month from earlier data.
        # At least 6 history months are required; seasonal model is only evaluated when enough data exists.
        candidate_names = ["Naive baseline", "Mean baseline", "Drift", "Holt damped trend"]
        if len(monthly) >= 24:
            candidate_names.append("Holt-Winters seasonal")
