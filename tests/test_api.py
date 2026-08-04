from fastapi.testclient import TestClient

from churn_service.app import app, get_model
from churn_service.training import build_pipeline, make_dataset


def test_health_reports_model_not_ready_when_artifact_is_missing(monkeypatch):
    get_model.cache_clear()
    monkeypatch.setattr("churn_service.app.load_model", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "model_not_ready"}
    get_model.cache_clear()


def test_prediction_contract(monkeypatch):
    features, target = make_dataset(rows=80)
    model = build_pipeline().fit(features, target)
    monkeypatch.setattr("churn_service.app.get_model", lambda: model)
    payload = features.iloc[0].to_dict()
    payload["tenure_months"] = int(payload["tenure_months"])
    payload["support_tickets"] = int(payload["support_tickets"])
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_risk"] in {"low", "high"}


def test_metrics_endpoint_is_available():
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert "churnguard_predictions_total" in response.text
