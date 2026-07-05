"""Financial metric functions."""
import numpy as np
import pandas as pd

from training.metrics import (
    calmar_ratio, directional_accuracy, max_drawdown,
    profit_factor, sharpe_ratio, win_rate,
)


def test_directional_accuracy_perfect_and_inverted():
    y = np.array([0.01, -0.02, 0.005, -0.001])
    assert directional_accuracy(y, y) == 1.0
    assert directional_accuracy(y, -y) == 0.0


def test_directional_accuracy_empty():
    assert directional_accuracy(np.array([]), np.array([])) == 0.0


def test_sharpe_zero_variance():
    assert sharpe_ratio(np.full(100, 0.001) * 0) == 0.0


def test_max_drawdown_known_value():
    curve = np.array([100.0, 120.0, 90.0, 130.0])
    # Peak 120 -> trough 90 = 25% drawdown
    assert abs(max_drawdown(curve) - 0.25) < 1e-12


def test_max_drawdown_monotonic_curve_is_zero():
    assert max_drawdown(np.array([1.0, 2.0, 3.0])) == 0.0


def test_win_rate_and_profit_factor():
    log = pd.DataFrame({'pnl': [100.0, -50.0, 200.0, -25.0]})
    assert win_rate(log) == 0.5
    assert abs(profit_factor(log) - 300.0 / 75.0) < 1e-12


def test_profit_factor_no_losses():
    log = pd.DataFrame({'pnl': [100.0, 50.0]})
    assert profit_factor(log) == float('inf')
