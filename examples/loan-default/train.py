"""Classification example: record a loan-default baseline in MLflow."""

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(7)
rows = 1_200
features = pd.DataFrame({
    "income": rng.lognormal(10.7, 0.45, rows),
    "debt_to_income": rng.uniform(0.02, 0.7, rows),
    "credit_history_years": rng.uniform(0, 25, rows),
    "late_payments": rng.poisson(1.1, rows),
})
log_odds = -3 + 3.8 * features.debt_to_income - 0.06 * features.credit_history_years + 0.35 * features.late_payments
target = rng.binomial(1, 1 / (1 + np.exp(-log_odds)))
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=.2, random_state=7, stratify=target)
model = LogisticRegression(max_iter=1_000).fit(x_train, y_train)
probabilities = model.predict_proba(x_test)[:, 1]
with mlflow.start_run(run_name="loan-default-baseline"):
    mlflow.log_params({"model": "logistic_regression", "rows": rows})
    mlflow.log_metrics({"roc_auc": roc_auc_score(y_test, probabilities), "average_precision": average_precision_score(y_test, probabilities)})
    mlflow.sklearn.log_model(model, "model")
print("Logged loan-default baseline to MLflow.")

