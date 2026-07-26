# Learning path

Work through one layer at a time. Commit after every milestone, and write a short note in your PR or README about what surprised you.

1. **Platform overview** — Start `docker compose up -d`, then identify each local service and the data it stores.
2. **Problem and data** — Read `training.py`. Explain why churn is a classification task and why a synthetic dataset makes this starter reproducible, not production-ready.
3. **Model** — Change one feature or classifier. Compare `accuracy` and `roc_auc` in MLflow; prefer ROC AUC here because churn can be imbalanced.
4. **Experiment tracking** — Inspect churn, loan-default, and demand-forecast runs. Explain why classification and regression use different metrics.
5. **Serving** — Read `schemas.py` before `app.py`. Send valid and invalid requests through `/docs` and understand API validation.
6. **Testing and CI** — Make the test fail on purpose, repair it, and inspect the GitHub Actions run after publishing.
7. **Containers and Kubernetes** — Build the image, deploy it on kind, and confirm `/health` and `/predict` work through port-forwarding.
8. **Production extensions** — Follow the staged additions in `platform-architecture.md`; do not add Kubeflow or KServe before the earlier layers work.

## Portfolio story

When presenting this project, say: “I built and deployed a reproducible churn-classification service. I treated the model pipeline as a versioned artifact, tracked experiments with MLflow, tested the API in CI, and packaged it for Kubernetes. Next, I am adding data versioning and monitoring.”
