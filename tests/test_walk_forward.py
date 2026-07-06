"""
The tests that would have caught the original target-leakage bug:
the training target must be the NEXT day's return, computed only from
data available at the cutoff.
"""
import numpy as np
import pandas as pd

from features.walk_forward import (
    FEATURE_COLUMNS, get_features_at, get_latest_features,
)


def test_target_is_next_day_return(ohlcv):
    cutoff = ohlcv.index[250].strftime('%Y-%m-%d')
    X, y, dates = get_features_at(ohlcv, cutoff)

    # Spot-check several rows: y[t] == close[t+1] / close[t] - 1
    for k in (0, 10, len(y) - 1):
        pos = ohlcv.index.get_loc(dates[k])
        expected = (ohlcv['Close'].iloc[pos + 1] /
                    ohlcv['Close'].iloc[pos] - 1.0)
        assert abs(y[k] - expected) < 1e-12


def test_cutoff_is_respected(ohlcv):
    cutoff = ohlcv.index[200].strftime('%Y-%m-%d')
    X, y, dates = get_features_at(ohlcv, cutoff)
    assert dates.max() <= pd.Timestamp(cutoff)

    latest = get_latest_features(ohlcv, cutoff)
    assert latest.index.max() <= pd.Timestamp(cutoff)


def test_training_rows_all_have_known_targets(ohlcv):
    """The last feature row (whose next day is unknown) must be dropped."""
    cutoff = ohlcv.index[-1].strftime('%Y-%m-%d')
    X, y, dates = get_features_at(ohlcv, cutoff)
    assert not np.isnan(y).any()
    # The very last date cannot appear: its next-day return doesn't exist
    assert dates.max() < ohlcv.index[-1]


def test_prediction_path_includes_cutoff_row(ohlcv):
    """Live prediction needs today's features - get_latest_features keeps them."""
    cutoff = ohlcv.index[-1].strftime('%Y-%m-%d')
    latest = get_latest_features(ohlcv, cutoff)
    assert latest.index[-1] == ohlcv.index[-1]
    assert list(latest.columns) == FEATURE_COLUMNS


def test_features_are_stationary_scale(ohlcv):
    """No feature should carry raw price levels (~150); everything is
    a ratio, return, or bounded oscillator."""
    cutoff = ohlcv.index[-1].strftime('%Y-%m-%d')
    X, _, _ = get_features_at(ohlcv, cutoff)
    assert np.abs(X).max() < 50, "a feature looks like a raw price level"
