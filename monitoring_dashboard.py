"""
Nexariza Model Monitoring Dashboard
Reads the FastAPI service's prediction log and MLflow tracking data to show
model performance and usage over time.

Run with: streamlit run monitoring_dashboard.py
"""

import streamlit as st
import pandas as pd
import json
import os
import requests

st.set_page_config(page_title="Nexariza Model Monitoring", page_icon="📊", layout="wide")

st.title("📊 Nexariza Model Monitoring Dashboard")
st.caption("Tracks the deployed churn prediction API: usage, latency, and prediction distribution.")

API_URL = st.text_input(
    "API base URL", value="http://localhost:8600",
    help="Point this at your deployed API (e.g. your Hugging Face Space URL) or leave as localhost for local testing.",
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Live API stats")
    if st.button("Fetch latest stats"):
        try:
            resp = requests.get(f"{API_URL}/stats", timeout=5)
            if resp.status_code == 200:
                stats = resp.json()
                st.metric("Total predictions", stats.get("total_predictions", 0))
                if stats.get("total_predictions", 0) > 0:
                    st.metric("Avg churn probability", f"{stats.get('avg_churn_probability', 0):.1%}")
                    st.metric("Avg latency (ms)", stats.get("avg_latency_ms", 0))
                    st.metric("High risk predictions", stats.get("high_risk_count", 0))
            else:
                st.error(f"API returned status {resp.status_code}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

with col2:
    st.subheader("Health check")
    if st.button("Check API health"):
        try:
            resp = requests.get(f"{API_URL}/health", timeout=5)
            if resp.status_code == 200:
                st.success(f"API is healthy: {resp.json()}")
            else:
                st.error(f"API returned status {resp.status_code}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

st.markdown("---")
st.subheader("Prediction history (from local log file)")

LOG_PATH = "prediction_log.jsonl"
if os.path.exists(LOG_PATH):
    with open(LOG_PATH) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if records:
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total logged predictions", len(df))
        c2.metric("Avg churn probability", f"{df['churn_probability'].mean():.1%}")
        c3.metric("Avg latency (ms)", f"{df['latency_ms'].mean():.1f}")

        st.line_chart(df.set_index("timestamp")["churn_probability"])
        st.dataframe(df[["timestamp", "churn_probability", "latency_ms"]], use_container_width=True)
    else:
        st.info("Log file exists but has no entries yet. Make a prediction via the API first.")
else:
    st.info("No prediction log found yet. Run the API and make at least one /predict request.")

st.markdown("---")
st.subheader("MLflow experiment tracking")
st.caption("Training run metrics logged during model development.")

try:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow_tracking/mlflow.db")
    runs = mlflow.search_runs(experiment_names=["nexariza-churn-predictor"])
    if not runs.empty:
        display_cols = [c for c in runs.columns if c.startswith("metrics.") or c.startswith("params.") or c == "run_id"]
        st.dataframe(runs[display_cols], use_container_width=True)
    else:
        st.info("No MLflow runs found. Run the training script first.")
except Exception as e:
    st.warning(f"Could not load MLflow tracking data: {e}")

st.markdown("---")
st.caption("Built for Nexariza AI | AI/ML Internship — Week 5 · FastAPI + Docker + MLflow")
