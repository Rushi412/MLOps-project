from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model.joblib"


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python -m churn_service.training` first.")
    return joblib.load(path)


def predict_probability(model, payload: dict) -> float:
    frame = pd.DataFrame([payload])
    return float(model.predict_proba(frame)[0, 1])

