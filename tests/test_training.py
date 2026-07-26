from churn_service.training import build_pipeline, make_dataset


def test_pipeline_trains_and_predicts_probability():
    features, target = make_dataset(rows=50)
    model = build_pipeline().fit(features, target)
    probability = model.predict_proba(features.iloc[:1])[0, 1]
    assert 0 <= probability <= 1

