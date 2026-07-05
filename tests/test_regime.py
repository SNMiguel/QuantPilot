"""Volatility regime detection and its size multiplier."""
import numpy as np
import pandas as pd

from features.regime import detect_regime


def _price_df(daily_vol, n=400, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, daily_vol, n)
    close = pd.Series(100 * np.cumprod(1 + rets))
    return pd.DataFrame({'Close': close})


def test_turbulent_scales_below_calm():
    calm = detect_regime(_price_df(0.006))
    # A regime whose recent vol is high vs its own history
    mixed = _price_df(0.006, n=300)
    spike = _price_df(0.03, n=100, seed=1)
    df = pd.DataFrame({'Close': pd.concat(
        [mixed['Close'], spike['Close'] + mixed['Close'].iloc[-1]],
        ignore_index=True)})
    turbulent = detect_regime(df)
    assert turbulent['size_multiplier'] <= calm['size_multiplier']


def test_multiplier_is_bounded():
    for vol in (0.004, 0.01, 0.05):
        r = detect_regime(_price_df(vol))
        assert 0.0 < r['size_multiplier'] <= 1.0
        assert r['regime'] in ('calm', 'normal', 'elevated', 'turbulent')


def test_insufficient_data_returns_safe_default():
    tiny = pd.DataFrame({'Close': [100.0, 101.0, 100.5]})
    r = detect_regime(tiny)
    assert r['size_multiplier'] == 1.0
    assert r['regime'] == 'normal'


def test_percentile_in_unit_interval(ohlcv):
    r = detect_regime(ohlcv)
    assert 0.0 <= r['percentile'] <= 1.0
