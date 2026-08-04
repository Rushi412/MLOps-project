import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from churn_service.model import load_model, predict_probability
from churn_service.schemas import ChurnRequest, PredictionResponse

app = FastAPI(title="ChurnGuard API", version="0.1.0")
MODEL_VERSION = os.getenv("MODEL_VERSION", "local")
PREDICTIONS = Counter("churnguard_predictions_total", "Predictions served", ["risk", "model_version"])
PREDICTION_PROBABILITY = Histogram(
    "churnguard_prediction_probability",
    "Distribution of churn probabilities",
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)


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


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ChurnRequest) -> PredictionResponse:
    try:
        probability = predict_probability(get_model(), request.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    risk = "high" if probability >= 0.5 else "low"
    PREDICTIONS.labels(risk=risk, model_version=MODEL_VERSION).inc()
    PREDICTION_PROBABILITY.observe(probability)
    return PredictionResponse(churn_probability=round(probability, 4), churn_risk=risk, model_version=MODEL_VERSION)

