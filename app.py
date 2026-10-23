import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from backend import prepare_monthly_data, compare_models, make_forecast

st.set_page_config(page_title="Monthly Demand Forecast", layout="wide")
st.title("Monthly Demand Forecasting")
st.write("Upload your sales dataset to analyze and forecast monthly demand.")

uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        monthly = prepare_monthly_data(df)

        st.subheader("Dataset Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Quantity", f"{monthly.sum():,.0f}")
        c2.metric("Average Monthly Demand", f"{monthly.mean():,.2f}")
        c3.metric("Months of Data", len(monthly))

        st.subheader("Historical Monthly Demand")
        st.line_chart(monthly)

        st.subheader("Model Comparison")
        scores = compare_models(monthly)

        if not scores.empty:
            st.dataframe(scores.round(2), use_container_width=True)
            best_model = scores.iloc[0]["Model"]
            st.success(f"Best model by MAE: {best_model}")
        else:
            best_model = "Simple Exponential Smoothing"
            st.info("Limited data: using Simple Exponential Smoothing.")

        periods = st.selectbox("Forecast months", [3, 6, 12], index=0)
        forecast = make_forecast(monthly, periods, best_model)

        result = pd.DataFrame({
            "Month": forecast.index.strftime("%Y-%m"),
            "Forecast_Quantity": forecast.round().astype(int).values
        })

        st.subheader("Future Demand Forecast")
        st.dataframe(result, use_container_width=True)
        st.metric("Total Forecast", f"{result['Forecast_Quantity'].sum():,}")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(monthly.index, monthly.values, marker="o", label="Historical")
        ax.plot(forecast.index, forecast.values, marker="o", label="Forecast")
        ax.set_xlabel("Month")
        ax.set_ylabel("Quantity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Download Forecast CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="monthly_demand_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a dataset with date and quantity columns.")
    st.caption("Supported columns include OrderDate and Quantity.")
