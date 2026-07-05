"""ATR-based position sizing and its hard exposure cap."""
import numpy as np
import pandas as pd

from risk.position_sizer import PositionSizer


def make_sizer():
    return PositionSizer(risk_per_trade=0.01, atr_multiplier=2.0,
                         max_position_pct=0.15)


def test_risk_based_size():
    # risk $ = 100k * 1% = 1000; stop = 2 * ATR(5) = 10 -> 100 shares
    shares = make_sizer().size(100_000.0, 50.0, atr=5.0)
    assert shares == 100


def test_exposure_cap_binds():
    # Tiny ATR would suggest a huge position; cap = 15% of 100k / $100 = 150
    shares = make_sizer().size(100_000.0, 100.0, atr=0.01)
    assert shares == 150


def test_invalid_inputs_return_zero():
    sizer = make_sizer()
    assert sizer.size(0.0, 100.0, 1.0) == 0
    assert sizer.size(100_000.0, 0.0, 1.0) == 0
    assert sizer.size(100_000.0, 100.0, 0.0) == 0


def test_size_multiplier_scales_down():
    sizer = make_sizer()
    full = sizer.size(100_000.0, 100.0, atr=0.01, size_multiplier=1.0)  # cap-bound = 150
    half = sizer.size(100_000.0, 100.0, atr=0.01, size_multiplier=0.5)
    assert half == 75
    assert half < full


def test_zero_multiplier_returns_zero():
    assert make_sizer().size(100_000.0, 100.0, atr=1.0, size_multiplier=0.0) == 0


def test_atr_positive_on_real_ranges(ohlcv):
    atr = make_sizer().calculate_atr(ohlcv)
    assert atr > 0


def test_atr_fallback_on_short_history(ohlcv):
    short = ohlcv.iloc[:5]
    atr = make_sizer().calculate_atr(short)
    expected = float(short['Close'].iloc[-1] * 0.01)
    assert abs(atr - expected) < 1e-9
