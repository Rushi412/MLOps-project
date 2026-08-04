from churn_service.monitoring import build_drift_report
from churn_service.training import make_dataset


def test_identical_distributions_are_stable():
    reference, _ = make_dataset(rows=500)
    report = build_drift_report(reference, reference.copy())
    assert report["status"] == "stable"
    assert report["drifted_features"] == []


def test_shifted_numeric_feature_is_detected():
    reference, _ = make_dataset(rows=500)
    current = reference.copy()
    current["monthly_charges"] = current["monthly_charges"] * 1.8
    report = build_drift_report(reference, current)
    assert "monthly_charges" in report["drifted_features"]
