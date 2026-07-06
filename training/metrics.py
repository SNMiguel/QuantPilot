"""
Financial performance metrics for evaluating trading strategy results.
Used by walk_forward_trainer (fold evaluation) and backtest/report.
"""
import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray,
                 risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio.

    Args:
        returns:        Daily return series (fractional, e.g. 0.01 = 1%).
        risk_free_rate: Annual risk-free rate (default 0.0).

    Returns:
        Sharpe ratio (float). Returns 0.0 if std is zero.
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess   = returns - daily_rf
    std      = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def directional_accuracy(y_true: np.ndarray,
                         y_pred: np.ndarray) -> float:
    """
    Fraction of predictions with the correct sign.

    The honest headline metric for a return forecaster: a coin flip
    scores ~0.50, so anything sustainably above ~0.53 on unseen data
    is meaningful. RMSE alone can look flattering while the model gets
    the direction wrong.

    Args:
        y_true: Realised returns.
        y_pred: Predicted returns.

    Returns:
        Accuracy in [0, 1]. Returns 0.0 for empty input.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum drawdown - largest peak-to-trough decline.

    Args:
        equity_curve: Array of portfolio values over time.

    Returns:
        Max drawdown as a positive fraction e.g. 0.15 = 15% drawdown.
    """
    if len(equity_curve) < 2:
        return 0.0
    peak     = np.maximum.accumulate(equity_curve)
    drawdown = np.where(peak > 0, (peak - equity_curve) / peak, 0.0)
    return float(np.max(drawdown))


def calmar_ratio(returns: np.ndarray,
                 equity_curve: np.ndarray) -> float:
    """
    Calmar ratio - annualised return divided by max drawdown.

    Returns 0.0 if max drawdown is zero.
    """
    annual_return = float(np.mean(returns) * 252)
    mdd           = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    return annual_return / mdd


def win_rate(trade_log: pd.DataFrame) -> float:
    """
    Fraction of trades that were profitable.

    Args:
        trade_log: DataFrame with a 'pnl' column.

    Returns:
        Win rate in [0, 1]. Returns 0.0 if trade_log is empty or lacks 'pnl'.
    """
    if trade_log.empty or 'pnl' not in trade_log.columns:
        return 0.0
    total = len(trade_log)
    if total == 0:
        return 0.0
    return float((trade_log['pnl'] > 0).sum() / total)


def profit_factor(trade_log: pd.DataFrame) -> float:
    """
    Gross profit divided by gross loss.

    Args:
        trade_log: DataFrame with a 'pnl' column.

    Returns:
        Profit factor (float). Returns 0.0 if no losing trades exist.
    """
    if trade_log.empty or 'pnl' not in trade_log.columns:
        return 0.0
    gross_profit = trade_log.loc[trade_log['pnl'] > 0, 'pnl'].sum()
    gross_loss   = abs(trade_log.loc[trade_log['pnl'] < 0, 'pnl'].sum())
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    # Simulate a modest equity curve
    returns      = np.random.normal(0.0005, 0.01, 252)
    equity_curve = 100_000 * np.cumprod(1 + returns)

    trade_log = pd.DataFrame({
        'pnl': np.random.normal(50, 200, 50)
    })

    preds = returns + np.random.normal(0, 0.01, 252)
    print(f"Dir. accuracy : {directional_accuracy(returns, preds):.4f}")
    print(f"Sharpe ratio  : {sharpe_ratio(returns):.4f}")
    print(f"Max drawdown  : {max_drawdown(equity_curve):.4f}")
    print(f"Calmar ratio  : {calmar_ratio(returns, equity_curve):.4f}")
    print(f"Win rate      : {win_rate(trade_log):.4f}")
    print(f"Profit factor : {profit_factor(trade_log):.4f}")
    print("training/metrics.py: OK")
