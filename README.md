# MLOps Learning Platform

An original, local-first MLOps platform for building several ML projects from the same production foundation. It follows the lifecycle demonstrated by larger MLOps platforms—experiment tracking, artifact storage, containerized serving, Kubernetes, and observability—while keeping every component understandable and free to run on a laptop.

## Platform architecture

```text
Examples (churn | loan default | demand forecast)
       |                         |
       +--> MLflow tracking <----+----> PostgreSQL metadata
                 |
                 +--> MinIO artifact store (models, metrics)
                              |
GitHub Actions -> Docker -> FastAPI serving -> Kubernetes (kind)
                                                |
                                      Prometheus -> Grafana
```

The components are deliberately layered. Start with the core workflow, then add the optional local platform services. See [platform architecture](docs/platform-architecture.md) and the [learning path](docs/learning-path.md).

## Example projects

| Example | ML task | What you learn |
| --- | --- | --- |
| `src/churn_service` | Customer-churn classification | API serving, pipelines, model probabilities |
| `examples/loan-default` | Loan-default classification | validation, classification metrics, experiment comparison |
| `examples/demand-forecast` | Retail demand regression | time-aware split, MAE, regression tracking |

All examples generate deterministic synthetic data. This makes the repository reproducible and prevents accidental use of sensitive customer data. Replace synthetic data only with a licensed, documented dataset.

## Start with one project

Install Python **3.11+** and Docker Desktop, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m churn_service.training
uvicorn churn_service.app:app --reload --reload-dir src --port 8000
```

Visit `http://127.0.0.1:8000/docs` to test the churn API. To run another example:

```powershell
python examples/loan-default/train.py
python examples/demand-forecast/train.py
```

## Start the local platform

```powershell
docker compose up -d
```

This starts MLflow (`http://localhost:5000`), MinIO (`http://localhost:9001`), PostgreSQL, Prometheus (`http://localhost:9090`), and Grafana (`http://localhost:3000`). The initial Grafana sign-in is `admin` / `admin`; change it immediately for any shared environment.

Set this before training if you want runs stored in the platform rather than local `mlruns/`:

```powershell
$env:MLFLOW_TRACKING_URI = 'http://localhost:5000'
$env:MLFLOW_S3_ENDPOINT_URL = 'http://localhost:9000'
$env:AWS_ACCESS_KEY_ID = 'minioadmin'
$env:AWS_SECRET_ACCESS_KEY = 'minioadmin'
$env:AWS_DEFAULT_REGION = 'ap-south-1'
```

## Run on local Kubernetes

Create a kind cluster, build the image, load it, then deploy:

```powershell
kind create cluster --config deploy/platform/kind-config.yaml
docker build -t churnguard:local .
kind load docker-image churnguard:local
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl port-forward service/churnguard 8080:80
```

The service is then available at `http://127.0.0.1:8080`. For a real registry, replace the local image with an immutable tag such as `ghcr.io/rushi412/mlops-learning-platform:1.0.0`.

## Checks

```powershell
ruff check .
pytest
docker build -t churnguard:local .
```

## Publish to GitHub

This folder is already initialized as a separate local Git repository. Create an empty `mlops-learning-platform` repository under your `Rushi412` GitHub account, then from this directory run:

```powershell
git add .
git commit -m "feat: initialize MLOps learning platform"
git remote add origin https://github.com/Rushi412/mlops-learning-platform.git
git push -u origin main
```

Set your future portfolio website in the GitHub repository's **About → Website** field and add its address to the `PORTFOLIO_URL` repository variable. The CI workflow checks that link after every change.
