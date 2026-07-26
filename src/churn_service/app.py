import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from churn_service.model import load_model, predict_probability
from churn_service.schemas import ChurnRequest, PredictionResponse

app = FastAPI(title="ChurnGuard API", version="0.1.0")
MODEL_VERSION = os.getenv("MODEL_VERSION", "local")


@lru_cache
def get_model():
    return load_model()


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_model()
        return {"status": "ok"}
    except FileNotFoundError:
        return {"status": "model_not_ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ChurnRequest) -> PredictionResponse:
    try:
        probability = predict_probability(get_model(), request.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return PredictionResponse(churn_probability=round(probability, 4), churn_risk="high" if probability >= 0.5 else "low", model_version=MODEL_VERSION)

