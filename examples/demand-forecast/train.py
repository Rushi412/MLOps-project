"""Regression example: log a simple, time-aware retail demand baseline."""

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

rng = np.random.default_rng(11)
days = pd.date_range("2024-01-01", periods=365, freq="D")
features = pd.DataFrame({"day_index": np.arange(len(days)), "weekday": days.dayofweek, "promotion": rng.binomial(1, .15, len(days))})
target = 100 + 14 * np.sin(features.day_index / 20) - 12 * (features.weekday >= 5) + 28 * features.promotion + rng.normal(0, 7, len(days))
split = 300  # Do not shuffle time-series data: future records remain in the test set.
model = RandomForestRegressor(n_estimators=150, random_state=11).fit(features.iloc[:split], target[:split])
predictions = model.predict(features.iloc[split:])
with mlflow.start_run(run_name="demand-forecast-baseline"):
    mlflow.log_params({"model": "random_forest", "n_estimators": 150, "train_end": str(days[split - 1].date())})
    mlflow.log_metrics({"mae": mean_absolute_error(target[split:], predictions), "rmse": mean_squared_error(target[split:], predictions) ** .5})
    mlflow.sklearn.log_model(model, "model")
print("Logged demand-forecast baseline to MLflow.")

