from churn_service.registry import meets_promotion_gate


def test_promotion_gate_accepts_equal_or_higher_metric():
    assert meets_promotion_gate(0.75, 0.70)
    assert meets_promotion_gate(0.70, 0.70)


def test_promotion_gate_rejects_lower_metric():
    assert not meets_promotion_gate(0.69, 0.70)
