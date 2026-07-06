"""
Leakage-free feature generation for walk-forward training and live prediction.

Two contracts:

  get_features_at(df, cutoff)      -> (X, y, dates) for TRAINING.
      y[t] is the NEXT-DAY simple return (close[t+1] / close[t] - 1).
      The final row is dropped because its next-day return is unknown.

  get_latest_features(df, cutoff)  -> feature DataFrame for PREDICTION.
      Includes the row at the cutoff date itself (no target needed).
      The caller predicts on the last row and interprets the model
      output as the expected next-day return.

Features are STATIONARY transforms of the raw indicators (ratios and
returns rather than price levels), so a model trained on one price
regime generalises to another and the target is honest: the model can
no longer reconstruct the current close from same-day price levels.
"""
import numpy as np
import pandas as pd
from features.indicators import add_indicators

# Ordered list of feature columns produced by _stationary_features().
# Keep in sync with the transform below - models are persisted against
# this exact layout (plus one trailing 'sentiment' column added later).
FEATURE_COLUMNS = [
    'close_ma5', 'close_ma10', 'close_ma20', 'close_ma50',
    'macd_norm', 'macd_hist_norm',
    'rsi',
    'bb_position', 'bb_width',
    'vol_ratio_5', 'vol_ratio_20',
    'ret_5', 'ret_10',
    'daily_return', 'volatility',
]


def _stationary_features(df_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw indicator columns into scale-free features.

    Args:
        df_ind: Output of add_indicators() - OHLCV plus 18 indicator columns.

    Returns:
        DataFrame with FEATURE_COLUMNS, NaN rows (indicator warmup) dropped.
    """
    close = df_ind['Close']
    out = pd.DataFrame(index=df_ind.index)

    # Price relative to its moving averages (percent above/below)
    out['close_ma5']  = close / df_ind['MA_5']  - 1.0
    out['close_ma10'] = close / df_ind['MA_10'] - 1.0
    out['close_ma20'] = close / df_ind['MA_20'] - 1.0
    out['close_ma50'] = close / df_ind['MA_50'] - 1.0

    # MACD scaled by price so it is comparable across price levels
    out['macd_norm']      = df_ind['MACD'] / close
    out['macd_hist_norm'] = (df_ind['MACD'] - df_ind['Signal_Line']) / close

    # RSI rescaled to [0, 1]
    out['rsi'] = df_ind['RSI'] / 100.0

    # Position within the Bollinger band and relative band width
    band = df_ind['BB_Upper'] - df_ind['BB_Lower']
    out['bb_position'] = np.where(band > 0,
                                  (close - df_ind['BB_Lower']) / band, 0.5)
    out['bb_width'] = np.where(df_ind['BB_Middle'] > 0,
                               band / df_ind['BB_Middle'], 0.0)

    # Volume relative to its own moving averages
    out['vol_ratio_5']  = df_ind['Volume'] / df_ind['Volume_MA_5']  - 1.0
    out['vol_ratio_20'] = df_ind['Volume'] / df_ind['Volume_MA_20'] - 1.0

    # Momentum expressed as returns, not dollar moves
    out['ret_5']  = close / close.shift(5)  - 1.0
    out['ret_10'] = close / close.shift(10) - 1.0

    out['daily_return'] = df_ind['Daily_Return']
    out['volatility']   = df_ind['Volatility']

    return out[FEATURE_COLUMNS].dropna()


def get_features_at(df: pd.DataFrame, cutoff_date: str):
    """
    Build the TRAINING matrix using only data up to and including cutoff_date.

    Args:
        df:          Raw OHLCV DataFrame with DatetimeIndex.
        cutoff_date: ISO date string e.g. '2023-06-01'. Only rows on or
                     before this date are used to compute indicators.

    Returns:
        X     (ndarray):       Feature matrix, shape (n_samples, n_features)
        y     (ndarray):       NEXT-DAY simple returns aligned with X
        dates (DatetimeIndex): Dates of the feature rows (the "today" of
                               each prediction, not the target day)
    """
    sliced   = df.loc[:cutoff_date].copy()
    features = _stationary_features(add_indicators(sliced))

    close = sliced['Close'].reindex(features.index)
    next_return = close.shift(-1) / close - 1.0

    # Drop the final row - its next-day return does not exist yet
    valid = next_return.notna()
    X     = features[valid].values
    y     = next_return[valid].values
    dates = features.index[valid]

    return X, y, dates


def get_latest_features(df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """
    Build the PREDICTION feature matrix up to and including cutoff_date.

    Unlike get_features_at(), the row at the cutoff date is kept - it is
    the row the caller predicts on. Returned as a DataFrame (date index,
    FEATURE_COLUMNS) so sentiment can be merged on before predicting.
    """
    sliced = df.loc[:cutoff_date].copy()
    return _stationary_features(add_indicators(sliced))


if __name__ == "__main__":
    # Synthetic OHLCV - no network required
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=300)
    close = pd.Series(150 * np.cumprod(1 + rng.normal(0.0005, 0.015, 300)),
                      index=dates)
    df = pd.DataFrame({
        'Open':   close.shift(1).fillna(close.iloc[0]),
        'High':   close * 1.01,
        'Low':    close * 0.99,
        'Close':  close,
        'Volume': rng.integers(50_000_000, 150_000_000, 300),
    })

    cutoff = dates[200].strftime('%Y-%m-%d')
    X, y, feat_dates = get_features_at(df, cutoff)

    assert feat_dates[-1] <= pd.Timestamp(cutoff), "Cutoff not respected"
    assert X.shape[1] == len(FEATURE_COLUMNS)

    # Target alignment: y[t] must equal close[t+1]/close[t] - 1
    t = feat_dates[10]
    pos = df.index.get_loc(t)
    expected = df['Close'].iloc[pos + 1] / df['Close'].iloc[pos] - 1.0
    assert abs(y[10] - expected) < 1e-12, "Target misaligned"

    latest = get_latest_features(df, cutoff)
    assert latest.index[-1] == pd.Timestamp(cutoff) or \
           latest.index[-1] <= pd.Timestamp(cutoff)

    print(f"Training matrix : {X.shape}, target = next-day return")
    print(f"Prediction rows : {latest.shape}, last date {latest.index[-1].date()}")
    print("features/walk_forward.py: OK")
