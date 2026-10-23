import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing, Holt, ExponentialSmoothing
)


def prepare_monthly_data(df):
    columns = {str(c).strip().lower(): c for c in df.columns}

    date_col = next(
        (columns[c] for c in ["orderdate", "order date", "date"] if c in columns),
        None
    )
    qty_col = next(
        (columns[c] for c in ["quantity", "qty", "demand", "sales"] if c in columns),
        None
    )

    if date_col is None or qty_col is None:
        raise ValueError("Date aur Quantity columns required hain.")

    data = df[[date_col, qty_col]].copy()
    data.columns = ["OrderDate", "Quantity"]
    data["OrderDate"] = pd.to_datetime(data["OrderDate"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data = data.dropna()

    if data.empty:
        raise ValueError("Dataset mein valid date/quantity data nahi hai.")

    monthly = (
        data.set_index("OrderDate")["Quantity"]
        .resample("MS").sum()
        .asfreq("MS", fill_value=0)
    )

    if len(monthly) < 3:
        raise ValueError("Kam se kam 3 months ka data chahiye.")

    return monthly


def compare_models(monthly):
    test_size = min(12, max(1, len(monthly) // 4))
    train = monthly.iloc[:-test_size]
    test = monthly.iloc[-test_size:]
    results = []

    models = {
        "Simple Exponential Smoothing": lambda: SimpleExpSmoothing(
            train, initialization_method="estimated"
        ).fit().forecast(len(test)),
        "Holt Trend": lambda: Holt(
            train, initialization_method="estimated"
        ).fit(optimized=True).forecast(len(test))
    }

    if len(train) >= 24:
        models["Holt-Winters Seasonal"] = lambda: ExponentialSmoothing(
            train, trend="add", seasonal="add",
            seasonal_periods=12, initialization_method="estimated"
        ).fit(optimized=True).forecast(len(test))

    for name, predict in models.items():
        try:
            pred = np.asarray(predict())
            results.append({
                "Model": name,
                "MAE": mean_absolute_error(test, pred),
                "RMSE": np.sqrt(mean_squared_error(test, pred))
            })
        except Exception:
            pass

    return pd.DataFrame(results).sort_values("MAE") if results else pd.DataFrame()


def make_forecast(monthly, periods=3, model_name=None):
    if model_name == "Holt-Winters Seasonal" and len(monthly) >= 24:
        model = ExponentialSmoothing(
            monthly, trend="add", seasonal="add",
            seasonal_periods=12, initialization_method="estimated"
        ).fit(optimized=True)
    elif model_name == "Holt Trend":
        model = Holt(
            monthly, initialization_method="estimated"
        ).fit(optimized=True)
    else:
        model = SimpleExpSmoothing(
            monthly, initialization_method="estimated"
        ).fit()

    forecast = model.forecast(periods).clip(lower=0)
    return forecast
