
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing




# Backend API URL
BACKEND_URL = "https://forecasting-project2-1.onrender.com"
FORECAST_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/forecast"
HEALTH_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/health"


warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Demand Intelligence | Forecast Studio",
    page_icon="📈",
    layout="wide",
)

# -------------------- Professional styling --------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1450px;
}
.hero {
    padding: 1.6rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #14213d, #1d3557, #287271);
    color: white;
    margin-bottom: 1.3rem;
}
.hero h1 {
    color: white;
    font-size: 2rem;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #e7eef7;
    margin: 0;
}
div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.18);
    padding: 14px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📈 Demand Intelligence — Forecast Studio</h1>
    <p>
    Enter historical monthly demand, compare forecasting methods,
    and plan future demand. No file upload required.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar settings --------------------
with st.sidebar:
    st.header("Forecast Settings")

    horizon = st.number_input(
        "Forecast horizon (months)",
        min_value=1,
        max_value=24,
        value=3,
        step=1,
    )

    st.caption("Choose how many future months to forecast.")

    st.divider()
    st.subheader("Data Entry Guidelines")

    st.markdown("""
    - Enter one demand value per month.
    - Use YYYY-MM date format.
    - Keep months consecutive.
    - At least 6 months of data are required.
    - 24+ months helps identify yearly seasonality.
    """)

    st.divider()
    st.caption(
        "Forecasts are estimates. Validate results against "
        "business conditions before making decisions."
    )

# -------------------- Historical data input --------------------
st.subheader("1. Historical Monthly Demand")

st.write(
    "Edit the example table below or replace it with your own "
    "historical monthly demand values. These starter values are "
    "illustrative only, not your actual business data."
)

latest_month = pd.Timestamp.today().to_period("M").to_timestamp()

default_dates = pd.date_range(
    end=latest_month,
    periods=12,
    freq="MS",
)

starter_data = pd.DataFrame({
    "Month": default_dates.strftime("%Y-%m"),
    "Demand": [
        120, 132, 128, 145, 151, 160,
        158, 172, 181, 176, 195, 208
    ],
})

if "history_editor" not in st.session_state:
    st.session_state["history_editor"] = starter_data.copy()

_, reset_column = st.columns([4, 1])

with reset_column:
    if st.button("Reset Example Data", use_container_width=True):
        st.session_state["history_editor"] = starter_data.copy()
        st.rerun()

edited_data = st.data_editor(
    st.session_state["history_editor"],
    key="history_table",
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Month": st.column_config.TextColumn(
            "Month (YYYY-MM)",
            help="Example: 2025-01",
            required=True,
        ),
        "Demand": st.column_config.NumberColumn(
            "Historical Demand",
            help="Units sold, orders, or monthly quantity",
            min_value=0.0,
            step=1.0,
            required=True,
        ),
    },
)

st.caption(
    "Add or delete rows using the table controls. "
    "Enter actual historical values for meaningful predictions."
)

# -------------------- Data validation --------------------
history = None
errors = []

if edited_data.empty:
    errors.append("Please enter at least 6 months of historical demand.")
else:
    data = edited_data.copy()

    data["Month"] = pd.to_datetime(
        data["Month"].astype(str).str.strip(),
        format="%Y-%m",
        errors="coerce",
    )

    data["Demand"] = pd.to_numeric(
        data["Demand"],
        errors="coerce",
    )

    if data["Month"].isna().any():
        errors.append(
            "Invalid month. Use YYYY-MM format, such as 2025-01."
        )

    if data["Demand"].isna().any():
        errors.append("Every demand value must be numeric.")

    if (data["Demand"].dropna() < 0).any():
        errors.append("Demand cannot be negative.")

    if not errors:
        data["Month"] = (
            data["Month"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        data = data.sort_values("Month")

        if data["Month"].duplicated().any():
            errors.append(
                "Duplicate months found. Keep only one row per month."
            )

        elif len(data) < 6:
            errors.append(
                "Please provide at least 6 valid monthly observations."
            )

        else:
            expected_index = pd.date_range(
                start=data["Month"].min(),
                end=data["Month"].max(),
                freq="MS",
            )

            if len(expected_index) != len(data):
                errors.append(
                    "Months must be consecutive. Add missing months "
                    "and enter actual demand. Use zero only when "
                    "demand was truly zero."
                )
            else:
                history = pd.Series(
                    data["Demand"].to_numpy(dtype=float),
                    index=expected_index,
                    name="Demand",
                )
                history.index.name = "Month"

for error in errors:
    st.warning(error)

# -------------------- Forecast functions --------------------
def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return float(mae), float(rmse)


def fit_holt_winters(series):
    if len(series) >= 24:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
    else:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )

    return model.fit(optimized=True)


def generate_forecast(series, steps):
    test_size = min(3, max(1, len(series) // 5))

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    model_results = []

    # Naive baseline
    naive_predictions = np.repeat(
        float(train.iloc[-1]),
        len(test),
    )

    mae, rmse = calculate_metrics(
        test.to_numpy(),
        naive_predictions,
    )

    model_results.append({
        "Model": "Naive Baseline",
        "MAE": mae,
        "RMSE": rmse,
    })

    # Holt-Winters validation
    if len(train) >= 6:
        try:
            fitted = fit_holt_winters(train)
            hw_predictions = fitted.forecast(len(test))

            mae, rmse = calculate_metrics(
                test.to_numpy(),
                hw_predictions.to_numpy(),
            )

            model_results.append({
                "Model": "Holt-Winters",
                "MAE": mae,
                "RMSE": rmse,
            })

        except Exception:
            pass

    scores = (
        pd.DataFrame(model_results)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    selected_model = scores.iloc[0]["Model"]

    # Fit selected model on all historical data
    try:
        if selected_model == "Holt-Winters":
            fitted = fit_holt_winters(series)
            predictions = np.asarray(
                fitted.forecast(int(steps)),
                dtype=float,
            )
        else:
            predictions = np.repeat(
                float(series.iloc[-1]),
                int(steps),
            )

    except Exception:
        selected_model = "Naive Baseline (Fallback)"
        predictions = np.repeat(
            float(series.iloc[-1]),
            int(steps),
        )

    future_dates = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1),
        periods=int(steps),
        freq="MS",
    )

    forecast = pd.Series(
        np.maximum(predictions, 0),
        index=future_dates,
        name="Forecast",
    )

    return forecast, scores, selected_model, test_size


# -------------------- Generate forecast --------------------
st.divider()
st.subheader("2. Generate Forecast")

if st.button(
    "✨ Analyze and Forecast",
    type="primary",
    use_container_width=True,
    disabled=history is None,
):
    try:
        forecast, scores, selected_model, test_size = (
            generate_forecast(history, horizon)
        )

        st.session_state["forecast_result"] = {
            "history": history.copy(),
            "forecast": forecast,
            "scores": scores,
            "model": selected_model,
            "test_size": test_size,
        }

    except Exception as exc:
        st.error(f"Forecast could not be generated: {exc}")

# -------------------- Display results --------------------
result = st.session_state.get("forecast_result")

if result is not None:
    same_history = (
        history is not None
        and history.index.equals(result["history"].index)
        and np.allclose(
            history.to_numpy(),
            result["history"].to_numpy(),
        )
    )

    if same_history:
        hist = result["history"]
        forecast = result["forecast"]
        scores = result["scores"]

        latest_demand = float(hist.iloc[-1])
        average_forecast = float(forecast.mean())
        total_forecast = float(forecast.sum())

        change_pct = (
            (average_forecast - latest_demand)
            / latest_demand * 100
            if latest_demand != 0
            else None
        )

        st.subheader("3. Forecast Results")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Selected Model",
            result["model"],
        )

        c2.metric(
            "Forecast Horizon",
            f"{len(forecast)} months",
        )

        c3.metric(
            "Total Forecast Demand",
            f"{total_forecast:,.1f}",
        )

        c4.metric(
            "Average Monthly Forecast",
            f"{average_forecast:,.1f}",
            (
                f"{change_pct:+.1f}% vs latest month"
                if change_pct is not None
                else None
            ),
        )

        # Chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist.values,
                mode="lines+markers",
                name="Historical Demand",
                line=dict(width=3, color="#287271"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode="lines+markers",
                name="Forecast",
                line=dict(
                    width=3,
                    dash="dash",
                    color="#f4a261",
                ),
            )
        )

        fig.add_vline(
            x=hist.index[-1],
            line_dash="dot",
            line_color="gray",
        )

        fig.update_layout(
            title="Historical Demand vs Future Forecast",
            xaxis_title="Month",
            yaxis_title="Demand",
            hovermode="x unified",
            height=450,
            margin=dict(l=20, r=20, t=70, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns([1.2, 1])

        forecast_table = pd.DataFrame({
            "Month": forecast.index.strftime("%Y-%m"),
            "Forecast Demand": np.round(
                forecast.values,
                2,
            ),
        })

        with left:
            st.subheader("Monthly Forecast")
            st.dataframe(
                forecast_table,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇ Download Forecast CSV",
                data=forecast_table.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="monthly_demand_forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with right:
            st.subheader("Model Comparison")

            displayed_scores = scores.copy()
            displayed_scores[["MAE", "RMSE"]] = (
                displayed_scores[["MAE", "RMSE"]].round(2)
            )

            st.dataframe(
                displayed_scores,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"Models were compared on the last "
                f"{result['test_size']} historical month(s). "
                "Lower MAE indicates better performance on this "
                "validation sample."
            )

        st.info(
            f"Selected **{result['model']}** based on historical "
            "validation performance. Consider promotions, stockouts, "
            "holidays, and other business changes when interpreting "
            "forecasts."
        )

    else:
        st.info(
            "The history table has changed. Click "
            "'Analyze and Forecast' again to refresh results."
        )

st.divider()

st.caption(
    "Demand Intelligence Forecast Studio | Manual monthly data entry "
    "| Naive baseline and Holt-Winters forecasting"
)
