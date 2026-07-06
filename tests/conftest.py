"""
Shared fixtures. Everything is synthetic - no network, no database,
no API keys required.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def ohlcv():
    """300 business days of synthetic OHLCV with a gentle upward drift."""
    rng   = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=300)
    close = pd.Series(
        150.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, 300)),
        index=dates,
    )
    return pd.DataFrame({
        'Open':   close.shift(1).fillna(close.iloc[0]),
        'High':   close * 1.008,
        'Low':    close * 0.992,
        'Close':  close,
        'Volume': rng.integers(50_000_000, 150_000_000, 300).astype(np.int64),
    })


@pytest.fixture
def empty_sentiment():
    return pd.DataFrame(columns=['score'])
