"""
Trains the churn model and logs the run to MLflow.
Run this from the mlflow_tracking/ directory: python train_and_log.py

This regenerates the same model shipped in models/, and logs its params and
metrics to a local MLflow tracking database (mlflow.db) for experiment tracking.
"""

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def build_synthetic_clients(n=1200, seed=42):
    np.random.seed(seed)
    tenure_months = np.random.gamma(shape=2.2, scale=9, size=n).clip(1, 60).astype(int)
    service_tiers = np.random.choice(["Basic", "Standard", "Premium", "Enterprise"], size=n, p=[0.30, 0.35, 0.25, 0.10])
    num_services_used = np.random.poisson(lam=3, size=n).clip(1, 8)
    industries = np.random.choice(
        ["E-commerce", "SaaS", "Healthcare", "Finance", "Real Estate", "Education", "Retail", "Consulting"], size=n
    )
    monthly_spend = np.round(
        (num_services_used * np.random.uniform(80, 220, size=n))
        + np.where(service_tiers == "Enterprise", 800, 0)
        + np.where(service_tiers == "Premium", 300, 0)
        + np.where(service_tiers == "Standard", 100, 0),
        2,
    )
    support_tickets_last_90d = np.random.poisson(lam=1.5, size=n)
    avg_response_time_hours = np.round(np.random.gamma(2, 6, size=n), 1)
    logins_last_30d = np.random.poisson(lam=12, size=n).clip(0, 60)
    nps_score = np.random.randint(0, 11, size=n)
    csat_score = np.round(np.random.uniform(1, 5, size=n), 1)
    onboarding_completed = np.random.choice([1, 0], size=n, p=[0.8, 0.2])
    had_late_payment = np.random.choice([1, 0], size=n, p=[0.18, 0.82])
    account_manager_assigned = np.random.choice([1, 0], size=n, p=[0.55, 0.45])

    engagement_score = (
        (logins_last_30d / 60) * 30
        + (nps_score / 10) * 25
        + (csat_score / 5) * 20
        + onboarding_completed * 10
        + account_manager_assigned * 10
        + (1 - np.clip(avg_response_time_hours / 48, 0, 1)) * 5
    )
    engagement_score = np.round(np.clip(engagement_score + np.random.normal(0, 6, size=n), 0, 100), 1)
    contract_type = np.random.choice(["Monthly", "Quarterly", "Annual"], size=n, p=[0.5, 0.3, 0.2])

    churn_logit = (
        -0.045 * tenure_months
        - 0.035 * engagement_score
        - 0.35 * num_services_used
        - 0.30 * nps_score
        - 0.55 * csat_score
        + 0.55 * support_tickets_last_90d
        + 1.1 * had_late_payment
        + np.where(contract_type == "Monthly", 1.0, 0.0)
        + np.where(contract_type == "Quarterly", 0.3, 0.0)
        - 0.8 * account_manager_assigned
        - 0.6 * onboarding_completed
        + 0.02 * avg_response_time_hours
        + 4.2
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = np.random.binomial(1, churn_prob)

    return pd.DataFrame({
        "industry": industries, "service_tier": service_tiers, "contract_type": contract_type,
        "tenure_months": tenure_months, "num_services_used": num_services_used, "monthly_spend": monthly_spend,
        "support_tickets_last_90d": support_tickets_last_90d, "avg_response_time_hours": avg_response_time_hours,
        "logins_last_30d": logins_last_30d, "nps_score": nps_score, "csat_score": csat_score,
        "onboarding_completed": onboarding_completed, "had_late_payment": had_late_payment,
        "account_manager_assigned": account_manager_assigned, "engagement_score": engagement_score,
        "churned": churned,
    })


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nexariza-churn-predictor")

    df = build_synthetic_clients()
    categorical_cols = ["industry", "service_tier", "contract_type"]
    X = pd.get_dummies(df.drop(columns=["churned"]), columns=categorical_cols, drop_first=True)
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.9, "colsample_bytree": 0.9,
    }
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    with mlflow.start_run(run_name="xgboost_churn_v1"):
        model = XGBClassifier(
            **params, scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }

        mlflow.log_param("model_type", "XGBClassifier")
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.xgboost.log_model(model, name="model")

        print("Run logged. Metrics:", metrics)
        print("Run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
