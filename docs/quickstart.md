# Five-minute local quickstart

This path runs the model workflow and API without starting the optional platform services.

## 1. Create the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## 2. Train the baseline

```powershell
python -m churn_service.training
```

This writes `artifacts/model.joblib` and logs an MLflow run locally.

## 3. Start the API

```powershell
uvicorn churn_service.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`, call `/health`, and submit a `/predict` request.

## 4. Validate the project

```powershell
ruff check .
pytest
```

## Optional: run the full platform

```powershell
docker compose up -d
```

The full stack adds MLflow, PostgreSQL, MinIO, Prometheus, and Grafana. Stop it with `docker compose down` when finished.
