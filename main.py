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

            raise HTTPException(
                status_code=400,
                detail="Use valid YYYY-MM months and numeric demand values.",
            )

        if not np.isfinite(df["demand"].to_numpy(dtype=float)).all():
            raise HTTPException(
                status_code=400,
                detail="Demand values must be finite numbers.",
            )

        if (df["demand"] < 0).any():
            raise HTTPException(
                status_code=400,
                detail="Demand cannot be negative.",
            )

        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
        df = df.sort_values("month").reset_index(drop=True)

        if df["month"].duplicated().any():
            raise HTTPException(
                status_code=400,
                detail="Duplicate months found. Keep one row per month.",
            )

        expected = pd.date_range(
            df["month"].min(),
            df["month"].max(),
            freq="MS",
        )

        if len(expected) != len(df) or not np.array_equal(
            df["month"].to_numpy(), expected.to_numpy()
        ):
            raise HTTPException(
                status_code=400,
                detail="Months must be consecutive. Add missing months.",
            )

        series = pd.Series(
            df["demand"].to_numpy(dtype=float),
            index=expected,
            name="Demand",
        )

        if request.history_months is not None:
            series = series.iloc[
                -min(request.history_months, len(series)):
            :]

        series.index.name = "Month"

        candidate_names = [
            "Naive baseline",
            "Mean baseline",
            "Drift",
            "Holt damped trend",
        ]

        if len(series) >= 30:
            candidate_names.append("Holt-Winters seasonal")

        test_count = min(6, max(1, len(series) - 5))
        first_test_index = len(series) - test_count
        scores = []

        for model_name in candidate_names:
            actual_values = []
            predicted_values = []

            for test_index in range(first_test_index, len(series)):
                training = series.iloc[:test_index]
                actual = float(series.iloc[test_index])

                try:
                    prediction = float(
                        fit_predict(model_name, training, 1)[0]
                    )
                    actual_values.append(actual)
                    predicted_values.append(prediction)
                except Exception:
                    continue

            if actual_values:
                mae = float(
                    mean_absolute_error(
                        actual_values, predicted_values
                    )
                )
                rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            actual_values, predicted_values
                        )
                    )
                )

                actual_array = np.asarray(actual_values, dtype=float)
                predicted_array = np.asarray(predicted_values, dtype=float)
                nonzero = actual_array != 0
                mape = None

                if nonzero.any():
                    mape = float(
                        np.mean(
                            np.abs(
                                (
                                    actual_array[nonzero]
                                    - predicted_array[nonzero]
                                )
                                / actual_array[nonzero]
                            )
                        ) * 100
                    )

                scores.append({
                    "model": model_name,
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2),
                    "MAPE": round(mape, 2) if mape is not None else None,
                    "test_points": len(actual_values),
                })

        if not scores:
            raise HTTPException(
                status_code=400,
                detail="Not enough valid history to compare forecasting models.",
            )

        scores.sort(key=lambda row: (row["MAE"], row["RMSE"]))
        best_model = scores[0]["model"]

        predictions = fit_predict(
            best_model,
            series,
            int(request.months),
        )

        future_dates = pd.date_range(
            start=series.index[-1] + pd.offsets.MonthBegin(1),
            periods=int(request.months),
            freq="MS",
        )

        forecast_rows = [
            {
                "month": date.strftime("%Y-%m"),
                "forecast": round(float(value), 2),
            }
            for date, value in zip(future_dates, predictions)
        ]

        return {
            "summary": {
                "months": int(len(series)),
                "total_quantity": round(float(series.sum()), 2),
                "average_demand": round(float(series.mean()), 2),
                "selected_model": best_model,
                "history_months_used": int(len(series)),
                "total_forecast_demand": round(
                    float(np.sum(predictions)), 2
                ),
                "average_monthly_forecast": round(
                    float(np.mean(predictions)), 2
                ),
                "validation_method": "rolling one-month-ahead backtest",
            },
            "best_model": best_model,
            "model_scores": scores,
            "history": [
                {
                    "month": date.strftime("%Y-%m"),
                    "quantity": round(float(value), 2),
                }
                for date, value in series.items()
            ],
            "forecast": forecast_rows,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not generate forecast: {str(exc)}",
        )
