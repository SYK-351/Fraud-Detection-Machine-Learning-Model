import numpy as np
from fraud.models import make_pipeline


def test_scaler_is_fit_on_training_fold_only():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [100.0, 101.0]])
    y = np.array([0, 1, 0])
    pipeline = make_pipeline("lr", "none")
    pipeline.fit(X[:2], y[:2])
    assert np.allclose(pipeline.named_steps["scale"].mean_, X[:2].mean(axis=0))
    assert not np.allclose(pipeline.named_steps["scale"].mean_, X.mean(axis=0))


def test_resampler_is_not_run_during_prediction():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0], [11.0, 12.0], [12.0, 13.0]])
    y = np.array([0] * 10 + [1] * 2)
    pipeline = make_pipeline("lr", "undersample").fit(X, y)
    before = pipeline.named_steps["model"].n_features_in_
    predictions = pipeline.predict_proba(X[:3])
    assert predictions.shape == (3, 2)
    assert pipeline.named_steps["model"].n_features_in_ == before
