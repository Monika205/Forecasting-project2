import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Monthly Demand Forecast",
    page_icon="📊",
    layout="wide"
)

st.title("Monthly Demand Forecasting")
st.write("Upload your dataset to analyze demand and forecast future quantities.")

# Render par deploy hone ke baad apna backend URL yahan set karein.
# Example: https://your-backend-name.onrender.com
API_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel dataset",
    type=["csv", "xlsx"]
)

months = st.selectbox(
    "Forecast horizon",
    [3, 6, 12],
    index=0
)

if st.button("Analyze and Forecast", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a dataset first.")
    else:
        try:
            with st.spinner("Analyzing data and generating forecast..."):
                response = requests.post(
                    f"{API_URL}/forecast",
                    params={"months": months},
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=180
                )

            if response.status_code != 200:
                try:
                    message = response.json().get("detail", response.text)
                except Exception:
                    message = response.text
                st.error(f"Backend error: {message}")
                st.stop()

            result = response.json()

            st.success("Forecast generated successfully!")

            summary = result["summary"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Quantity", f'{summary["total_quantity"]:,.0f}')
            c2.metric("Average Monthly Demand", f'{summary["average_demand"]:,.2f}')
            c3.metric("Months of Data", summary["months"])

            history = pd.DataFrame(result["history"])
            forecast = pd.DataFrame(result["forecast"])

            st.subheader("Historical Demand and Forecast")

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(
                history["month"], history["quantity"],
                marker="o", label="Historical demand"
            )
            ax.plot(
                forecast["month"], forecast["quantity"],
                marker="o", linestyle="--", label="Forecast"
            )
            ax.set_xlabel("Month")
            ax.set_ylabel("Quantity")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Model Comparison")
            scores = pd.DataFrame(result["model_scores"])
            if not scores.empty:
                st.dataframe(scores, use_container_width=True)

            st.info(f'Selected model: {result["best_model"]}')

            st.subheader("Future Demand Forecast")
            st.dataframe(forecast, use_container_width=True)

            st.download_button(
                "Download Forecast CSV",
                data=forecast.to_csv(index=False).encode("utf-8"),
                file_name="monthly_demand_forecast.csv",
                mime="text/csv"
            )

        except requests.exceptions.RequestException as e:
            st.error(
                "Backend se connection nahi ho paaya. "
                "Check karein ki backend chal raha hai aur BACKEND_URL sahi hai."
            )
            st.caption(str(e))
