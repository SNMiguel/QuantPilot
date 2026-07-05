"""
Conformal prediction intervals on the ensemble.

Checks the two properties that make them useful: (1) empirical coverage on
unseen data is near the nominal level, and (2) the zero-exclusion gate is
consistent with the interval bounds.
"""
import numpy as np
import pytest

from models.ensemble import EnsembleModel, make_base_models


@pytest.fixture
def fitted_model():
    rng = np.random.default_rng(1)
    n, f = 800, 15
    X = rng.normal(size=(n, f))
    y = 0.004 * X[:, 0] + 0.002 * X[:, 1] + rng.normal(0, 0.01, n)
    m = EnsembleModel(make_base_models())
    m.fit(X[:600], y[:600], n_splits=5)
    return m, X[600:], y[600:]


def test_halfwidth_monotonic_in_coverage(fitted_model):
    m, _, _ = fitted_model
    assert m.conformal_halfwidth(0.80) <= m.conformal_halfwidth(0.95)


def test_empirical_coverage_near_nominal(fitted_model):
    m, X_test, y_test = fitted_model
    _, lo, hi = m.predict_interval(X_test, coverage=0.80)
    covered = np.mean((y_test >= lo) & (y_test <= hi))
    # Split-conformal guarantees marginal coverage up to finite-sample slack;
    # allow a band around the 0.80 target.
    assert 0.70 <= covered <= 0.92


def test_interval_excludes_zero_matches_bounds(fitted_model):
    m, X_test, _ = fitted_model
    point, lo, hi = m.predict_interval(X_test, coverage=0.80)
    gate = m.interval_excludes_zero(X_test, coverage=0.80)
    expected = (lo > 0) | (hi < 0)
    assert np.array_equal(gate, expected)


def test_intervals_require_fit():
    m = EnsembleModel(make_base_models())
    with pytest.raises(RuntimeError):
        m.conformal_halfwidth(0.8)
