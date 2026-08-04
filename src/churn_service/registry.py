"""MLflow model registration and controlled alias promotion."""

import argparse

import mlflow
from mlflow import MlflowClient

DEFAULT_MODEL_NAME = "churnguard"
CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"


def meets_promotion_gate(metric_value: float, minimum: float) -> bool:
    """Return whether a measured validation metric passes the promotion gate."""
    return metric_value >= minimum


def register_candidate(run_id: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Register a run's logged model and point the candidate alias at it."""
    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, CANDIDATE_ALIAS, result.version)
    return str(result.version)


def promote_candidate(
    model_name: str = DEFAULT_MODEL_NAME,
    metric_name: str = "roc_auc",
    minimum: float = 0.70,
) -> str:
    """Promote the candidate alias to champion only when its run metric passes."""
    client = MlflowClient()
    candidate = client.get_model_version_by_alias(model_name, CANDIDATE_ALIAS)
    metric_value = client.get_run(candidate.run_id).data.metrics.get(metric_name)
    if metric_value is None:
        raise ValueError(f"Candidate run does not contain required metric: {metric_name}")
    if not meets_promotion_gate(metric_value, minimum):
        raise ValueError(
            f"Candidate {metric_name}={metric_value:.4f} is below promotion gate {minimum:.4f}"
        )
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, candidate.version)
    return str(candidate.version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register or promote a ChurnGuard model")
    parser.add_argument("--run-id", help="MLflow run ID containing a model artifact")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--minimum-roc-auc", type=float, default=0.70)
    args = parser.parse_args()

    if args.run_id:
        print(f"Registered candidate version {register_candidate(args.run_id, args.model_name)}")
    if args.promote:
        version = promote_candidate(args.model_name, minimum=args.minimum_roc_auc)
        print(f"Promoted champion version {version}")
    if not args.run_id and not args.promote:
        parser.error("provide --run-id, --promote, or both")


if __name__ == "__main__":
    main()
