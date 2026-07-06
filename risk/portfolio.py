"""
Portfolio state tracker.
Syncs with the live Alpaca account and enforces exposure limits
before any order is submitted.
"""
import config
from training.metrics import max_drawdown
import numpy as np


class Portfolio:
    """
    Tracks live portfolio state and enforces risk limits.
    Requires an AlpacaFeed and Database instance.
    """

    def __init__(self, feed, db):
        """
        Args:
            feed: AlpacaFeed instance (for live account data).
            db:   Database instance (for historical snapshots).
        """
        self.feed = feed
        self.db   = db

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def get_open_positions(self) -> dict:
        """
        Fetch current open positions from Alpaca.

        Returns:
            dict keyed by ticker:
            { 'AAPL': {'qty': 10, 'market_value': 2250.0}, ... }
        """
        try:
            positions = self.feed.api.list_positions()
            return {
                p.symbol: {
                    'qty':          int(float(p.qty)),
                    'market_value': float(p.market_value),
                }
                for p in positions
            }
        except Exception as e:
            print(f"WARN: Could not fetch positions: {e}")
            return {}

    def get_portfolio_value(self) -> float:
        """
        Return current total equity from Alpaca account.

        Returns:
            Portfolio equity as float. Returns 0.0 on error.
        """
        try:
            return self.feed.get_account()['equity']
        except Exception as e:
            print(f"WARN: Could not fetch portfolio value: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def is_within_limits(self, ticker: str,
                         shares: int,
                         price: float) -> bool:
        """
        Check if adding `shares` of `ticker` at `price` keeps the
        position within MAX_POSITION_PCT of total portfolio equity.

        Args:
            ticker: Stock symbol.
            shares: Number of shares to add.
            price:  Current price per share.

        Returns:
            True if the trade is within limits, False otherwise.
        """
        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            return False

        # Existing position in this ticker
        positions       = self.get_open_positions()
        existing_value  = positions.get(ticker, {}).get('market_value', 0.0)
        new_trade_value = shares * price
        total_exposure  = existing_value + new_trade_value

        pct = total_exposure / portfolio_value

        if pct > config.MAX_POSITION_PCT:
            print(f"  WARN: Risk limit: {ticker} exposure would be "
                  f"{pct*100:.1f}% > {config.MAX_POSITION_PCT*100:.0f}% cap")
            return False

        return True

    # ------------------------------------------------------------------
    # Snapshots & drawdown
    # ------------------------------------------------------------------

    def snapshot(self, date: str) -> None:
        """
        Record today's portfolio equity to the database.

        Args:
            date: ISO date string 'YYYY-MM-DD'.
        """
        value = self.get_portfolio_value()
        self.db.snapshot_portfolio(date, value)
        print(f"  Portfolio snapshot: ${value:,.2f}  ({date})")

    def get_max_drawdown(self) -> float:
        """
        Compute maximum drawdown from historical portfolio snapshots.

        Returns:
            Max drawdown as a positive fraction (e.g. 0.12 = 12%).
            Returns 0.0 if fewer than 2 snapshots exist.
        """
        history = self.db.get_portfolio_history()
        if len(history) < 2:
            return 0.0
        equity_curve = history['value'].values
        return max_drawdown(equity_curve)


if __name__ == "__main__":
    import config
    from data.alpaca_feed import AlpacaFeed
    from data.database import Database

    feed = AlpacaFeed(config.ALPACA_API_KEY,
                      config.ALPACA_SECRET_KEY,
                      config.ALPACA_BASE_URL)
    db   = Database(config.DB_URL)

    portfolio = Portfolio(feed, db)

    print("Open positions  :", portfolio.get_open_positions())
    print("Portfolio value : $", f"{portfolio.get_portfolio_value():,.2f}")

    # Test limit check - 10 shares of AAPL at $200
    within = portfolio.is_within_limits("AAPL", 10, 200.0)
    print(f"Within limits (10 x $200): {within}")

    # Snapshot today
    from datetime import date
    portfolio.snapshot(date.today().isoformat())

    print("risk/portfolio.py: OK")
