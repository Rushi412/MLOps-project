"""Small, dependency-light feature drift report for the learning platform."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn_service.training import make_dataset

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "support_tickets"]
CATEGORICAL_FEATURES = ["contract_type", "internet_service"]


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate PSI using reference quantiles and stable epsilon smoothing."""
    edges = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy(dtype=float))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    epsilon = 1e-6
    ref_share = np.clip(ref_counts / max(ref_counts.sum(), 1), epsilon, None)
    cur_share = np.clip(cur_counts / max(cur_counts.sum(), 1), epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def categorical_distance(reference: pd.Series, current: pd.Series) -> float:
    """Calculate total-variation distance between categorical distributions."""
    categories = sorted(set(reference.astype(str)) | set(current.astype(str)))
    ref_share = reference.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
    cur_share = current.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
    return float(0.5 * np.abs(ref_share - cur_share).sum())


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_threshold: float = 0.20,
    categorical_threshold: float = 0.15,
) -> dict:
    """Build a JSON-serializable feature report with explicit thresholds."""
    features = {}
    for name in NUMERIC_FEATURES:
        score = population_stability_index(reference[name], current[name])
        features[name] = {"method": "psi", "score": round(score, 6), "drifted": score >= numeric_threshold}
    for name in CATEGORICAL_FEATURES:
        score = categorical_distance(reference[name], current[name])
        features[name] = {"method": "total_variation", "score": round(score, 6), "drifted": score >= categorical_threshold}
    drifted = sorted(name for name, result in features.items() if result["drifted"])
    return {
        "status": "drift_detected" if drifted else "stable",
        "drifted_features": drifted,
        "thresholds": {"numeric_psi": numeric_threshold, "categorical_distance": categorical_threshold},
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local feature drift report")
    parser.add_argument("--output", type=Path, default=Path("reports/drift-report.json"))
    parser.add_argument("--simulate-shift", action="store_true")
    args = parser.parse_args()

    reference, _ = make_dataset(rows=1_000)
    current, _ = make_dataset(rows=1_000)
    if args.simulate_shift:
        current = current.copy()
        current["monthly_charges"] = current["monthly_charges"] * 1.35
        current["support_tickets"] = current["support_tickets"] + 3
    report = build_drift_report(reference, current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with status={report['status']}")


if __name__ == "__main__":
    main()
