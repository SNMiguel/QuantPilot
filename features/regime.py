"""
Volatility regime detection.

Position sizing that ignores the volatility environment risks the same
dollar exposure in a calm tape and a crash. This classifies the current
regime by comparing recent realized volatility to its own trailing
distribution, and returns a size multiplier that shrinks exposure when
volatility spikes.

The classifier is intentionally simple and transparent (a rolling-vol
percentile), not a hidden-state model - it needs no training, can't
silently break, and its output is easy to reason about on the dashboard.
A Gaussian-HMM version is a natural future upgrade; the interface here
(detect_regime -> dict with a size_multiplier) would not change.
"""
import numpy as np
import pandas as pd

# Regime thresholds on the trailing percentile of realized volatility,
# and the exposure multiplier applied in each. Turbulent markets get
# roughly half size; calm/normal markets trade full size.
_TURBULENT_PCT = 0.85
_ELEVATED_PCT  = 0.70
_MULTIPLIER = {
    'calm':      1.00,
    'normal':    1.00,
    'elevated':  0.65,
    'turbulent': 0.40,
}


def detect_regime(df: pd.DataFrame,
                  vol_window: int = 20,
                  history_window: int = 252) -> dict:
    """
    Classify the current volatility regime from an OHLCV DataFrame.

    Args:
        df:             OHLCV with a 'Close' column, chronological.
        vol_window:     Lookback (days) for realized volatility.
        history_window: Trailing window the current vol is ranked against.

    Returns:
        dict:
            regime          : 'calm' | 'normal' | 'elevated' | 'turbulent'
            annualized_vol  : current realized vol, annualized (float)
            percentile      : where current vol sits in its history [0,1]
            size_multiplier : exposure scale in (0, 1]
    """
    default = {
        'regime': 'normal',
        'annualized_vol': 0.0,
        'percentile': 0.5,
        'size_multiplier': 1.0,
    }
    if df is None or 'Close' not in df or len(df) < vol_window + 2:
        return default

    returns = df['Close'].pct_change().dropna()
    realized = returns.rolling(vol_window).std() * np.sqrt(252)
    realized = realized.dropna()
    if realized.empty:
        return default

    current = float(realized.iloc[-1])
    history = realized.iloc[-history_window:]
    # Percentile of the current value within its own recent history
    percentile = float((history <= current).mean())

    if percentile >= _TURBULENT_PCT:
        regime = 'turbulent'
    elif percentile >= _ELEVATED_PCT:
        regime = 'elevated'
    elif percentile <= 0.25:
        regime = 'calm'
    else:
        regime = 'normal'

    return {
        'regime': regime,
        'annualized_vol': round(current, 4),
        'percentile': round(percentile, 3),
        'size_multiplier': _MULTIPLIER[regime],
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 380 calm days, then a sharp 20-day volatility spike at the end so the
    # most recent window sits unambiguously in the top percentile.
    calm_rets  = rng.normal(0, 0.006, 380)
    spike_rets = rng.normal(0, 0.035, 20)
    close = pd.Series(100 * np.cumprod(1 + np.concatenate([calm_rets, spike_rets])))
    df = pd.DataFrame({'Close': close})

    calm = detect_regime(df.iloc[:380])
    turb = detect_regime(df)
    print(f"Calm window : {calm}")
    print(f"After spike : {turb}")
    assert turb['size_multiplier'] < calm['size_multiplier']
    assert turb['regime'] in ('elevated', 'turbulent')
    print("features/regime.py: OK")
