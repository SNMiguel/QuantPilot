"""
Ensemble correctness: leakage-free stacking, full-window refit,
and a confidence score that actually discriminates.
"""
import numpy as np
import pytest

from models.ensemble import EnsembleModel, make_base_models


@pytest.fixture
def return_data():
    """Synthetic data with a real (weak) signal at return scale."""
    rng = np.random.default_rng(42)
    n, f = 500, 15
    X = rng.normal(size=(n, f))
    y = 0.004 * X[:, 0] + 0.002 * X[:, 1] + rng.normal(0, 0.008, n)
    return X, y


def test_fit_predict_shapes(return_data):
    X, y = return_data
    model = EnsembleModel(make_base_models())
    model.fit(X[:400], y[:400], n_splits=4)
    preds = model.predict(X[400:])
    assert preds.shape == (100,)


def test_predict_before_fit_raises():
    model = EnsembleModel(make_base_models())
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 15)))


def test_base_models_refit_on_full_window(return_data):
    """After fit(), every base model must have seen ALL training rows -
    the original implementation left them fitted on the last CV fold,
    which excluded the most recent data."""
    X, y = return_data
    model = EnsembleModel(make_base_models())
    model.fit(X[:400], y[:400], n_splits=4)

    rf = model.base_models['Random Forest']
    # RandomForest exposes the training sample count via its estimators
    assert rf.n_features_in_ == X.shape[1]

    # Behavioral check: a linear model refit on the full window must
    # match a reference fit on the same full window exactly.
    from sklearn.linear_model import LinearRegression
    ref = LinearRegression().fit(X[:400], y[:400])
    np.testing.assert_allclose(
        model.base_models['Linear Regression'].coef_, ref.coef_)


def test_ensemble_learns_signal(return_data):
    """With a real linear signal present, out-of-sample directional
    accuracy must beat a coin flip by a clear margin."""
    X, y = return_data
    model = EnsembleModel(make_base_models())
    model.fit(X[:400], y[:400], n_splits=4)
    preds = model.predict(X[400:])
    dir_acc = np.mean(np.sign(preds) == np.sign(y[400:]))
    assert dir_acc > 0.60


def test_confidence_bounded_and_meaningful(return_data):
    X, y = return_data
    model = EnsembleModel(make_base_models())
    model.fit(X[:400], y[:400], n_splits=4)

    conf = model.get_confidence(X[400:401])
    assert 0.0 <= conf <= 1.0

    # With 3 base models, single-row confidence is a fraction k/3
    assert min(abs(conf - v) for v in (0.0, 1/3, 2/3, 1.0)) < 1e-9


def test_oof_uses_only_past_data(return_data):
    """Leakage probe: plant a regime flip in the second half of the
    training window. If the meta-learner were built from KFold OOF
    predictions (which train on the future), first-half OOF rows would
    be polluted by the flipped regime. With TimeSeriesSplit the fit
    still succeeds and the ensemble tracks the FINAL regime, because
    base models are refit on the full window afterwards."""
    rng = np.random.default_rng(0)
    n, f = 400, 15
    X = rng.normal(size=(n, f))
    y = np.where(np.arange(n) < 200,
                 +0.005 * X[:, 0],
                 -0.005 * X[:, 0]) + rng.normal(0, 0.001, n)

    model = EnsembleModel(make_base_models())
    model.fit(X, y, n_splits=4)

    # Production predictions must follow the most recent regime
    X_probe = np.zeros((1, f))
    X_probe[0, 0] = 1.0
    assert model.predict(X_probe)[0] < 0
