"""
Nexariza ML Model Serving System
FastAPI app serving the Week 2 churn prediction model as a REST API.

Run locally with: uvicorn main:app --reload
Swagger docs available at /docs once running.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import time
import json
import os
from datetime import datetime

app = FastAPI(
    title="Nexariza Churn Prediction API",
    description="Predicts client churn risk and returns the top factors driving the prediction.",
    version="1.0.0",
)

MODEL = joblib.load("models/nexariza_churn_model.pkl")
MODEL_COLUMNS = joblib.load("models/model_columns.pkl")
RECOMMENDATION_MAP = joblib.load("models/recommendation_map.pkl")

CATEGORICAL_COLS = ["industry", "service_tier", "contract_type"]
LOG_PATH = "prediction_log.jsonl"


class ClientInput(BaseModel):
    industry: Literal["E-commerce", "SaaS", "Healthcare", "Finance", "Real Estate", "Education", "Retail", "Consulting"]
    service_tier: Literal["Basic", "Standard", "Premium", "Enterprise"]
    contract_type: Literal["Monthly", "Quarterly", "Annual"]
    tenure_months: int = Field(..., ge=0, le=120, description="Months as a client")
    num_services_used: int = Field(..., ge=0, le=20)
    monthly_spend: float = Field(..., ge=0)
    support_tickets_last_90d: int = Field(..., ge=0)
    avg_response_time_hours: float = Field(..., ge=0)
    logins_last_30d: int = Field(..., ge=0)
    nps_score: int = Field(..., ge=0, le=10)
    csat_score: float = Field(..., ge=1, le=5)
    onboarding_completed: bool
    had_late_payment: bool
    account_manager_assigned: bool
    engagement_score: float = Field(..., ge=0, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "industry": "SaaS",
                "service_tier": "Premium",
                "contract_type": "Monthly",
                "tenure_months": 4,
                "num_services_used": 2,
                "monthly_spend": 650.0,
                "support_tickets_last_90d": 5,
                "avg_response_time_hours": 30.0,
                "logins_last_30d": 3,
                "nps_score": 3,
                "csat_score": 2.1,
                "onboarding_completed": False,
                "had_late_payment": True,
                "account_manager_assigned": False,
                "engagement_score": 22.0,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str
    top_recommendations: list[str]
    latency_ms: float


def build_feature_row(client: ClientInput) -> pd.DataFrame:
    raw = client.dict()
    raw["onboarding_completed"] = int(raw["onboarding_completed"])
    raw["had_late_payment"] = int(raw["had_late_payment"])
    raw["account_manager_assigned"] = int(raw["account_manager_assigned"])

    row = pd.DataFrame([raw])
    row = pd.get_dummies(row, columns=CATEGORICAL_COLS, drop_first=True)
    for col in MODEL_COLUMNS:
        if col not in row.columns:
            row[col] = 0
    return row[MODEL_COLUMNS]


def get_recommendations(feature_row: pd.DataFrame, top_n: int = 3) -> list[str]:
    # Simple approach: flag which known risk-factor columns are in an unfavorable range
    # for this specific client, using the same recommendation map built in Week 2.
    recs = []
    row = feature_row.iloc[0]
    checks = [
        ("engagement_score", row.get("engagement_score", 100) < 40),
        ("tenure_months", row.get("tenure_months", 100) < 6),
        ("nps_score", row.get("nps_score", 10) < 5),
        ("csat_score", row.get("csat_score", 5) < 3),
        ("support_tickets_last_90d", row.get("support_tickets_last_90d", 0) > 3),
        ("had_late_payment", row.get("had_late_payment", 0) == 1),
        ("account_manager_assigned", row.get("account_manager_assigned", 1) == 0),
        ("onboarding_completed", row.get("onboarding_completed", 1) == 0),
    ]
    for key, triggered in checks:
        if triggered and key in RECOMMENDATION_MAP:
            recs.append(RECOMMENDATION_MAP[key])
        if len(recs) >= top_n:
            break
    if not recs:
        recs = ["No major risk factors flagged for this client."]
    return recs


def log_prediction(client: ClientInput, prob: float, latency_ms: float):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": client.dict(),
        "churn_probability": prob,
        "latency_ms": latency_ms,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/")
def root():
    return {
        "service": "Nexariza Churn Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientInput):
    start = time.time()
    try:
        feature_row = build_feature_row(client)
        prob = float(MODEL.predict_proba(feature_row)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    latency_ms = round((time.time() - start) * 1000, 2)

    if prob >= 0.6:
        risk_level = "High"
    elif prob >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    recommendations = get_recommendations(feature_row)
    log_prediction(client, prob, latency_ms)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        risk_level=risk_level,
        top_recommendations=recommendations,
        latency_ms=latency_ms,
    )


@app.get("/stats")
def stats():
    """Basic monitoring stats pulled from the prediction log."""
    if not os.path.exists(LOG_PATH):
        return {"total_predictions": 0}

    with open(LOG_PATH) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    if not lines:
        return {"total_predictions": 0}

    probs = [l["churn_probability"] for l in lines]
    latencies = [l["latency_ms"] for l in lines]

    return {
        "total_predictions": len(lines),
        "avg_churn_probability": round(sum(probs) / len(probs), 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "high_risk_count": sum(1 for p in probs if p >= 0.6),
    }
