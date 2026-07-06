"""
Event-driven backtester.

Iterates over historical dates chronologically. At each step:
  1. Compute features using only data up to that date (no leakage) -
     the same get_latest_features() path the live daily job uses
  2. Predict the next-day return and generate a signal
  3. Simulate fill at the NEXT day's open price
  4. Apply commission and slippage
  5. Track portfolio value, cash, and position

Returns an equity curve DataFrame that backtest/report.py uses
to compute financial metrics.
"""
import numpy as np
import pandas as pd

from features.walk_forward import get_latest_features
from features.sentiment_features import merge_sentiment
from features.regime import detect_regime


class BacktestEngine:
    """Chronological event-driven backtester."""

    def __init__(self, commission_per_share: float = 0.01,
                 initial_capital: float = 100_000.0,
                 stop_loss_pct: float = 0.02,
                 slippage_bps: float = 5.0):
        """
        Args:
            commission_per_share: Cost per share traded (default $0.01).
            initial_capital:      Starting portfolio cash (default $100,000).
            stop_loss_pct:        Exit long position if price drops this far
                                  below entry (default 2%).
            slippage_bps:         Adverse fill assumption in basis points -
                                  buys fill above the open, sells below.
        """
        self.commission_per_share = commission_per_share
        self.initial_capital      = initial_capital
        self.stop_loss_pct        = stop_loss_pct
        self.slippage_bps         = slippage_bps

    def run(self, df: pd.DataFrame,
            sentiment_df: pd.DataFrame,
            model,
            signal_gen,
            sizer) -> pd.DataFrame:
        """
        Run the full backtest.

        Args:
            df:           Raw OHLCV DataFrame with DatetimeIndex.
                          Must cover at least 60 rows (rolling window warmup).
            sentiment_df: Sentiment scores DataFrame (date index, 'score' col).
                          Can be empty.
            model:        Predicts NEXT-DAY RETURNS: .predict(X) -> ndarray.
                          If it has .get_confidence(X), that is used;
                          otherwise confidence defaults to 0.7.
            signal_gen:   SignalGenerator instance.
            sizer:        PositionSizer instance.

        Returns:
            DataFrame with columns:
                date, portfolio_value, cash, position_qty,
                trade_side, trade_qty, trade_price
            One row per trading day in the backtest window.
        """
        dates  = df.index
        n      = len(dates)
        warmup = 60   # indicator warmup (MA_50) plus a small margin

        cash     = self.initial_capital
        position = 0       # shares held (long only)
        entry_px = 0.0     # price at which current position was opened
        slip     = self.slippage_bps / 10_000.0

        records = []

        for i in range(warmup, n - 1):
            today     = dates[i]
            tomorrow  = dates[i + 1]
            today_str = today.strftime('%Y-%m-%d')

            # Features up to and including today (no future leakage)
            feature_df = get_latest_features(df, today_str)
            if feature_df.empty:
                continue
            feature_df = merge_sentiment(feature_df, sentiment_df)

            # Predict next-day return on today's row
            X_today = feature_df.values[-1:].reshape(1, -1)
            predicted_return = float(model.predict(X_today)[0])

            if hasattr(model, 'get_confidence'):
                confidence = model.get_confidence(X_today)
            else:
                confidence = 0.7

            current_price = float(df.loc[today, 'Close'])
            open_price    = float(df.loc[tomorrow, 'Open'])

            signal = signal_gen.generate_from_return(
                current_price, predicted_return, confidence)
            action = signal['signal']

            # Stop-loss overrides signal
            if position > 0 and entry_px > 0:
                if current_price <= entry_px * (1 - self.stop_loss_pct):
                    action = 'SELL'

            trade_side = ''
            trade_qty  = 0
            trade_px   = 0.0

            # --- Execute signal ---
            if action == 'BUY' and position == 0:
                fill_price = open_price * (1 + slip)
                atr        = sizer.calculate_atr(df.loc[:today])
                pv         = cash + position * current_price
                regime     = detect_regime(df.loc[:today])
                qty        = sizer.size(pv, current_price, atr,
                                        size_multiplier=regime['size_multiplier'])

                if qty > 0:
                    cost       = qty * fill_price
                    commission = qty * self.commission_per_share

                    if cost + commission <= cash:
                        cash      -= (cost + commission)
                        position  += qty
                        entry_px   = fill_price
                        trade_side = 'BUY'
                        trade_qty  = qty
                        trade_px   = fill_price

            elif action == 'SELL' and position > 0:
                fill_price = open_price * (1 - slip)
                qty        = position
                proceeds   = qty * fill_price
                commission = qty * self.commission_per_share

                cash      += proceeds - commission
                position   = 0
                entry_px   = 0.0
                trade_side = 'SELL'
                trade_qty  = qty
                trade_px   = fill_price

            portfolio_value = cash + position * open_price

            records.append({
                'date':            tomorrow,
                'portfolio_value': round(portfolio_value, 2),
                'cash':            round(cash, 2),
                'position_qty':    position,
                'trade_side':      trade_side,
                'trade_qty':       trade_qty,
                'trade_price':     round(trade_px, 4),
            })

        equity_curve = pd.DataFrame(records)
        if not equity_curve.empty:
            equity_curve.set_index('date', inplace=True)

        return equity_curve


if __name__ == "__main__":
    print("BacktestEngine tested via jobs/backtest_job.py and tests/")
    print("backtest/engine.py: OK")
