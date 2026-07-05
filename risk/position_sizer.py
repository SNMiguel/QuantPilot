"""
Volatility-based position sizer using Average True Range (ATR).

Formula:
    raw_shares = (portfolio_value * risk_per_trade) / (atr * atr_multiplier)
    cap_shares  = (portfolio_value * max_position_pct) / current_price
    shares      = floor(min(raw_shares, cap_shares))

This ensures:
  - Each trade risks at most risk_per_trade % of the portfolio
  - No single position exceeds max_position_pct of the portfolio
"""
import numpy as np
import pandas as pd


class PositionSizer:
    """ATR-based position sizing with a hard portfolio exposure cap."""

    def __init__(self, risk_per_trade: float = 0.01,
                 atr_multiplier: float = 2.0,
                 max_position_pct: float = None):
        """
        Args:
            risk_per_trade:   Fraction of portfolio risked per trade (default 1%).
            atr_multiplier:   Stop-loss width in ATR units (default 2×ATR).
            max_position_pct: Hard cap — single position as fraction of portfolio.
                              Defaults to config.MAX_POSITION_PCT (0.15).
        """
        import config
        self.risk_per_trade   = risk_per_trade
        self.atr_multiplier   = atr_multiplier
        self.max_position_pct = max_position_pct if max_position_pct is not None \
                                else config.MAX_POSITION_PCT

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range over the last `period` rows.

        True Range = max(High−Low, |High−PrevClose|, |Low−PrevClose|)

        Args:
            df:     OHLCV DataFrame with DatetimeIndex.
                    Must have at least period+1 rows.
            period: Lookback window (default 14).

        Returns:
            ATR as a float. Returns a fallback of 1% of last close if
            the DataFrame is too short.
        """
        if len(df) < period + 1:
            # Fallback: 1% of last close price
            return float(df['Close'].iloc[-1] * 0.01)

        high  = df['High'].values
        low   = df['Low'].values
        close = df['Close'].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:]  - close[:-1]),
            )
        )
        return float(tr[-period:].mean())

    # ------------------------------------------------------------------
    # Size
    # ------------------------------------------------------------------

    def size(self, portfolio_value: float,
             current_price: float,
             atr: float,
             size_multiplier: float = 1.0) -> int:
        """
        Calculate the number of shares to trade.

        Args:
            portfolio_value: Total equity in the account.
            current_price:   Current price of the asset.
            atr:             ATR value from calculate_atr().
            size_multiplier: Scales the final share count (e.g. a volatility
                             regime multiplier in (0, 1]). Applied to both
                             the ATR-based size and the exposure cap so a
                             turbulent-market trade is smaller everywhere.

        Returns:
            Integer number of shares (floored). Returns 0 if any
            input is invalid or the result rounds to zero.
        """
        if portfolio_value <= 0 or current_price <= 0 or atr <= 0:
            return 0
        if size_multiplier <= 0:
            return 0

        # ATR-based: risk $ / stop-loss $
        risk_dollars  = portfolio_value * self.risk_per_trade
        stop_loss_per = atr * self.atr_multiplier
        raw_shares    = risk_dollars / stop_loss_per

        # Portfolio cap
        max_dollars = portfolio_value * self.max_position_pct
        cap_shares  = max_dollars / current_price

        shares = int(min(raw_shares, cap_shares) * size_multiplier)
        return max(shares, 0)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    sizer = PositionSizer(risk_per_trade=0.01, atr_multiplier=2.0,
                          max_position_pct=0.15)

    # Simulate 30 days of OHLCV data around $200
    np.random.seed(42)
    n = 30
    close  = 200 + np.cumsum(np.random.randn(n))
    high   = close + np.random.uniform(0.5, 2.0, n)
    low    = close - np.random.uniform(0.5, 2.0, n)
    volume = np.random.randint(50_000_000, 150_000_000, n)

    df = pd.DataFrame({'Open': close, 'High': high, 'Low': low,
                       'Close': close, 'Volume': volume})

    atr = sizer.calculate_atr(df)
    print(f"ATR (14-day)         : ${atr:.4f}")

    portfolio_value = 100_000.0
    current_price   = float(close[-1])
    shares          = sizer.size(portfolio_value, current_price, atr)
    position_value  = shares * current_price

    print(f"Portfolio value      : ${portfolio_value:,.0f}")
    print(f"Current price        : ${current_price:.2f}")
    print(f"Shares to trade      : {shares}")
    print(f"Position value       : ${position_value:,.2f}")
    print(f"Portfolio exposure   : {position_value/portfolio_value*100:.1f}%")
    print("risk/position_sizer.py: OK")
