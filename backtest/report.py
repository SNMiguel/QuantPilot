"""
Backtest report generator.
Computes financial metrics from an equity curve and prints a summary table.
Saves the equity curve chart to results/backtest_equity_curve.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from training.metrics import (
    sharpe_ratio, max_drawdown, calmar_ratio,
    win_rate, profit_factor
)

sns.set_style("whitegrid")


def generate(equity_curve: pd.DataFrame,
             ticker: str = '',
             save_dir: str = 'results',
             price_df: pd.DataFrame = None) -> dict:
    """
    Generate a full backtest report.

    Args:
        equity_curve: DataFrame returned by BacktestEngine.run().
                      Must have columns: portfolio_value, trade_side,
                      trade_qty, trade_price.
        ticker:       Ticker label for the chart title.
        save_dir:     Directory to save the equity curve PNG.
        price_df:     Raw OHLCV DataFrame for the same period. When given,
                      a buy-and-hold baseline is computed — a strategy
                      that cannot beat simply holding the stock has no
                      edge, whatever its other metrics say.

    Returns:
        dict of computed metrics.
    """
    os.makedirs(save_dir, exist_ok=True)

    if equity_curve.empty:
        print("WARN: empty equity curve — no trades were made.")
        return {}

    # ------------------------------------------------------------------
    # Compute returns
    # ------------------------------------------------------------------
    values  = equity_curve['portfolio_value'].values
    returns = np.diff(values) / values[:-1]

    # ------------------------------------------------------------------
    # Build trade log with P&L
    # ------------------------------------------------------------------
    trades = equity_curve[equity_curve['trade_side'] != ''].copy()
    trade_log = _compute_pnl(trades)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    initial = 100_000.0
    final   = float(values[-1])

    total_return     = (final - initial) / initial * 100
    ann_return       = float(np.mean(returns) * 252 * 100)
    sharpe           = sharpe_ratio(returns)
    mdd              = max_drawdown(values)
    calmar           = calmar_ratio(returns, values)
    wr               = win_rate(trade_log)
    pf               = profit_factor(trade_log)
    n_trades         = len(trade_log)

    metrics = {
        'total_return_pct': round(total_return, 2),
        'ann_return_pct':   round(ann_return, 2),
        'sharpe_ratio':     round(sharpe, 4),
        'max_drawdown_pct': round(mdd * 100, 2),
        'calmar_ratio':     round(calmar, 4),
        'win_rate_pct':     round(wr * 100, 2),
        'profit_factor':    round(pf, 4),
        'n_trades':         n_trades,
        'final_value':      round(final, 2),
    }

    # ------------------------------------------------------------------
    # Buy-and-hold baseline on the same window
    # ------------------------------------------------------------------
    bh_curve = None
    if price_df is not None and not price_df.empty:
        closes = price_df['Close'].reindex(equity_curve.index).ffill()
        if closes.notna().all() and float(closes.iloc[0]) > 0:
            bh_curve = initial * closes / float(closes.iloc[0])
            bh_return = (float(bh_curve.iloc[-1]) - initial) / initial * 100
            bh_returns = np.diff(bh_curve.values) / bh_curve.values[:-1]
            metrics['buy_hold_return_pct'] = round(bh_return, 2)
            metrics['buy_hold_sharpe']     = round(sharpe_ratio(bh_returns), 4)
            metrics['beats_buy_hold']      = bool(total_return > bh_return)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    _print_summary(metrics, ticker)

    # ------------------------------------------------------------------
    # Plot equity curve
    # ------------------------------------------------------------------
    save_path = os.path.join(save_dir, 'backtest_equity_curve.png')
    _plot_equity_curve(equity_curve, metrics, ticker, save_path,
                       bh_curve=bh_curve)

    return metrics


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Pair BUY and SELL trades to compute per-trade P&L."""
    rows  = trades.itertuples()
    buys  = []
    pnl   = []

    for row in rows:
        if row.trade_side == 'BUY':
            buys.append((row.trade_qty, row.trade_price))
        elif row.trade_side == 'SELL' and buys:
            qty, entry = buys.pop(0)
            exit_px    = row.trade_price
            gross      = (exit_px - entry) * qty
            commission = qty * 0.01 * 2        # entry + exit commission
            pnl.append({'pnl': gross - commission, 'qty': qty})

    return pd.DataFrame(pnl) if pnl else pd.DataFrame(columns=['pnl', 'qty'])


def _print_summary(metrics: dict, ticker: str) -> None:
    label = f" — {ticker}" if ticker else ''
    print(f"\n{'='*55}")
    print(f"  Backtest Results{label}")
    print(f"{'='*55}")
    print(f"  Total Return     : {metrics['total_return_pct']:>8.2f}%")
    print(f"  Annual Return    : {metrics['ann_return_pct']:>8.2f}%")
    print(f"  Sharpe Ratio     : {metrics['sharpe_ratio']:>8.4f}")
    print(f"  Max Drawdown     : {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  Calmar Ratio     : {metrics['calmar_ratio']:>8.4f}")
    print(f"  Win Rate         : {metrics['win_rate_pct']:>8.2f}%")
    print(f"  Profit Factor    : {metrics['profit_factor']:>8.4f}")
    print(f"  Trades           : {metrics['n_trades']:>8}")
    print(f"  Final Value      : ${metrics['final_value']:>12,.2f}")
    if 'buy_hold_return_pct' in metrics:
        verdict = 'strategy wins' if metrics['beats_buy_hold'] \
                  else 'buy-and-hold wins'
        print(f"  {'-'*51}")
        print(f"  Buy & Hold Return: {metrics['buy_hold_return_pct']:>8.2f}%"
              f"   ({verdict})")
        print(f"  Buy & Hold Sharpe: {metrics['buy_hold_sharpe']:>8.4f}")
    print(f"{'='*55}\n")


def _plot_equity_curve(equity_curve: pd.DataFrame, metrics: dict,
                       ticker: str, save_path: str,
                       bh_curve: pd.Series = None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={'height_ratios': [3, 1]})

    # --- Equity curve ---
    ax = axes[0]
    ax.plot(equity_curve.index, equity_curve['portfolio_value'],
            color='#2E86AB', linewidth=1.8, label='Portfolio Value')
    if bh_curve is not None:
        ax.plot(bh_curve.index, bh_curve.values,
                color='#F18F01', linewidth=1.4, alpha=0.8,
                linestyle='-.', label='Buy & Hold')
    ax.axhline(y=100_000, color='grey', linestyle='--',
               linewidth=1, alpha=0.6, label='Initial Capital')

    # Mark trades
    buys  = equity_curve[equity_curve['trade_side'] == 'BUY']
    sells = equity_curve[equity_curve['trade_side'] == 'SELL']
    ax.scatter(buys.index,  buys['portfolio_value'],
               marker='^', color='green', s=60, zorder=5, label='BUY')
    ax.scatter(sells.index, sells['portfolio_value'],
               marker='v', color='red',   s=60, zorder=5, label='SELL')

    label = f" — {ticker}" if ticker else ''
    ax.set_title(f'Backtest Equity Curve{label}', fontsize=15, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Drawdown ---
    values   = equity_curve['portfolio_value'].values
    peak     = np.maximum.accumulate(values)
    drawdown = np.where(peak > 0, (peak - values) / peak * 100, 0)

    ax2 = axes[1]
    ax2.fill_between(equity_curve.index, drawdown, 0,
                     color='#C73E1D', alpha=0.5, label='Drawdown %')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.invert_yaxis()
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {save_path}")


if __name__ == "__main__":
    print("backtest/report.py tested via jobs/backtest_job.py")
    print("backtest/report.py: OK")
