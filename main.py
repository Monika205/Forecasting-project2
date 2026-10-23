
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    Holt,
    SimpleExpSmoothing,
)
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
            raise HTTPException(
                status_code=400,
                detail="At least 6 months of data required.",
            )

        df = pd.DataFrame(
            [row.model_dump() for row in request.data]
        )

        df["month"] = pd.to_datetime(
            df["month"], format="%Y-%m", errors="coerce"
        )
        df["demand"] = pd.to_numeric(
            df["demand"], errors="coerce"
        )

        if df["month"].isna().any() or df["demand"].isna().any():
            raise HTTPException(
                status_code=400,
                detail="Use valid YYYY-MM months and numeric demand.",
            )

        if (df["demand"] < 0).any():
            raise HTTPException(
                status_code=400,
                detail="Demand cannot be negative.",
            )

        df["month"] = (
            df["month"].dt.to_period("M").dt.to_timestamp()
        )
        df = df.sort_values("month").reset_index(drop=True)

        if df["month"].duplicated().any():
            raise HTTPException(
                status_code=400,
                detail="Duplicate months found. Keep one row per month.",
            )

        expected = pd.date_range(
            start=df["month"].min(),
            end=df["month"].max(),
            freq="MS",
        )

        if not df["month"].equals(
            pd.Series(expected, name="month")
        ):
            raise HTTPException(
                status_code=400,
                detail="Months must be consecutive. Add missing months.",
            )

        monthly = pd.Series(
            df["demand"].to_numpy(dtype=float),
            index=expected,
            name="Demand",
        )
        monthly.index.name = "Month"

        test_size = min(3, max(1, len(monthly) // 5))
        train = monthly.iloc[:-test_size]
        test = monthly.iloc[-test_size:]

        scores = []

        # Naive baseline model
        naive_pred = np.repeat(
            float(train.iloc[-1]), len(test)
        )

        scores.append({
            "model": "Naive baseline",
            "MAE": round(
                float(mean_absolute_error(test, naive_pred)), 2
            ),
            "RMSE": round(
                float(np.sqrt(mean_squared_error(test, naive_pred))),
                2,
            ),
        })

        # Holt trend model
        if len(train) >= 6:
            try:
                fitted_holt = Holt(
                    train,
                    initialization_method="estimated",
                ).fit()

                pred = np.asarray(
                    fitted_holt.forecast(len(test)),
                    dtype=float,
                )

                scores.append({
                    "model": "Holt Trend",
                    "MAE": round(
                        float(mean_absolute_error(test, pred)), 2
                    ),
                    "RMSE": round(
                        float(np.sqrt(mean_squared_error(test, pred))),
                        2,
                    ),
                })
            except Exception:
                pass

        # Seasonal model needs sufficient training history
        if len(train) >= 24:
            try:
                fitted_seasonal = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=12,
                    initialization_method="estimated",
                ).fit()

                pred = np.asarray(
                    fitted_seasonal.forecast(len(test)),
                    dtype=float,
                )

                scores.append({
                    "model": "Holt-Winters Seasonal",
                    "MAE": round(
                        float(mean_absolute_error(test, pred)), 2
                    ),
                    "RMSE": round(
                        float(np.sqrt(mean_squared_error(test, pred))),
                        2,
                    ),
                })
            except Exception:
                pass

        scores.sort(key=lambda row: row["MAE"])
        best = scores[0]["model"]

        # Fit the selected model on all available data
        if best == "Holt-Winters Seasonal" and len(monthly) >= 24:
            fitted = ExponentialSmoothing(
                monthly,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit()

            predictions = fitted.forecast(request.months)

        elif best == "Holt Trend":
            fitted = Holt(
                monthly,
                initialization_method="estimated",
            ).fit()

            predictions = fitted.forecast(request.months)

        else:
            fitted = SimpleExpSmoothing(
                monthly,
                initialization_method="estimated",
            ).fit()

            predictions = fitted.forecast(request.months)

        predictions = np.maximum(
            np.asarray(predictions, dtype=float), 0
        )

        future_dates = pd.date_range(
            start=monthly.index[-1] + pd.offsets.MonthBegin(1),
            periods=request.months,
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
                "months": int(len(monthly)),
                "total_quantity": float(monthly.sum()),
                "average_demand": round(float(monthly.mean()), 2),
                "selected_model": best,
                "total_forecast_demand": round(
                    float(np.sum(predictions)), 2
                ),
                "average_monthly_forecast": round(
                    float(np.mean(predictions)), 2
                ),
            },
            "best_model": best,
            "model_scores": scores,
            "history": [
                {
                    "month": date.strftime("%Y-%m"),
                    "quantity": float(value),
                }
                for date, value in monthly.items()
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
