"""Train a reproducible baseline model and log the experiment to MLflow."""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model.joblib"


def make_dataset(rows: int = 1_000) -> tuple[pd.DataFrame, pd.Series]:
    """Create a realistic, deterministic data set for learning and automated tests."""
    rng = np.random.default_rng(RANDOM_STATE)
    frame = pd.DataFrame({
        "tenure_months": rng.integers(0, 73, rows),
        "monthly_charges": rng.uniform(20, 120, rows).round(2),
        "support_tickets": rng.poisson(2, rows),
        "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], rows, p=[.55, .25, .20]),
        "internet_service": rng.choice(["dsl", "fiber", "none"], rows, p=[.35, .50, .15]),
    })
    frame["total_charges"] = (frame.tenure_months * frame.monthly_charges).round(2)
    risk = (-2.2 + 0.8 * (frame.contract_type == "month-to-month")
            + 0.55 * (frame.internet_service == "fiber") + 0.18 * frame.support_tickets
            - 0.035 * frame.tenure_months + rng.normal(0, 0.65, rows))
    target = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-risk))), name="churn")
    return frame, target


def build_pipeline() -> Pipeline:
    numeric = ["tenure_months", "monthly_charges", "total_charges", "support_tickets"]
    categorical = ["contract_type", "internet_service"]
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1_000))])


def train() -> dict[str, float]:
    features, target = make_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, probabilities >= 0.5), "roc_auc": roc_auc_score(y_test, probabilities)}
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)
    with mlflow.start_run() as run:
        mlflow.log_params({"algorithm": "logistic_regression", "random_state": RANDOM_STATE, "train_rows": len(x_train)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_artifact(str(ARTIFACT_PATH), artifact_path="portable-artifact")
        if os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true":
            from churn_service.registry import register_candidate

            register_candidate(run.info.run_id)
    return metrics


if __name__ == "__main__":
    result = train()
    print(f"Saved model to {ARTIFACT_PATH}; ROC AUC: {result['roc_auc']:.3f}")

