# Platform architecture and boundaries

This repository provides the **application-side** MLOps foundation. It is not a claim that every production platform can be installed with one command; high-cost or cluster-heavy systems are introduced only after the core workflow is understood.

| Layer | Local component | Purpose | Portfolio evidence |
| --- | --- | --- | --- |
| Experiment tracking | MLflow + PostgreSQL | parameters, metrics, model history | comparable MLflow runs |
| Artifact store | MinIO | model and experiment artifacts | versioned model artifact |
| Serving | FastAPI + Docker | validated prediction API | OpenAPI page and container |
| Deployment | kind + Kubernetes | repeatable local rollout | manifest and readiness probe |
| Observability | Prometheus + Grafana | platform monitoring starting point | local dashboards |
| Automation | GitHub Actions | linting, tests, portfolio-link check | green CI badge |

## What comes next

After all three examples run locally, add one capability at a time:

1. DVC with a remote store for real dataset versions.
2. MLflow model registry and an explicit staging-to-production promotion policy.
3. Prometheus metrics in the FastAPI service: request count, latency, prediction distribution, and model version.
4. Data-quality and drift checks (for example, Evidently) before automated retraining.
5. Kubeflow Pipelines or Prefect only when local scripts become hard to coordinate.
6. KServe only after you are comfortable with Kubernetes, since it adds substantial cluster complexity.

This staged approach mirrors a full MLOps platform without hiding the fundamentals under too much infrastructure.

