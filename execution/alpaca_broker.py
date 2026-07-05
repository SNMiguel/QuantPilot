"""
Alpaca paper trading broker wrapper.
Submits orders, checks positions, and queries market status.
All calls go to the paper trading endpoint (ALPACA_BASE_URL).
"""
import alpaca_trade_api as tradeapi


class AlpacaBroker:
    """Thin wrapper over the Alpaca REST API for order execution."""

    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2',
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(self, ticker: str, qty: int, side: str,
                     order_type: str = 'market') -> dict:
        """
        Submit an order to Alpaca.

        Args:
            ticker:     Stock symbol e.g. 'AAPL'.
            qty:        Number of shares (must be > 0).
            side:       'buy' or 'sell' (lowercase).
            order_type: 'market' (default) or 'limit'.

        Returns:
            Alpaca order object as a dict.

        Raises:
            Exception on API error — let OrderManager catch and log.
        """
        if qty <= 0:
            raise ValueError(f"Order qty must be > 0, got {qty}")

        order = self.api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force='day',
        )

        return {
            'id':         order.id,
            'ticker':     order.symbol,
            'side':       order.side,
            'qty':        int(order.qty),
            'type':       order.type,
            'status':     order.status,
            'created_at': str(order.created_at),
        }

    def cancel_all_orders(self) -> None:
        """Cancel all open orders. Called at the start of each daily job."""
        try:
            self.api.cancel_all_orders()
            print("  Stale orders cleared.")
        except Exception as e:
            print(f"  WARN: Could not cancel orders: {e}")

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> dict | None:
        """
        Return current held position for a ticker, or None if flat.

        Returns:
            dict with keys: ticker, qty, market_value, avg_entry_price
            or None if no position exists.
        """
        try:
            p = self.api.get_position(ticker)
            return {
                'ticker':          p.symbol,
                'qty':             int(float(p.qty)),
                'market_value':    float(p.market_value),
                'avg_entry_price': float(p.avg_entry_price),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Market status
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Return True if the US equity market is currently open."""
        try:
            return self.api.get_clock().is_open
        except Exception as e:
            print(f"  WARN: could not check market status: {e}")
            return False

    def market_traded_today(self) -> bool:
        """
        Return True if today is (or was) a trading day.

        The daily job runs AFTER the 4 PM ET close, when is_market_open()
        is always False — using the clock there would skip every session.
        The calendar endpoint answers the question actually being asked:
        "did the market trade today?" Market orders submitted after the
        close queue for the next session's open.
        """
        try:
            from datetime import date
            today = date.today().isoformat()
            calendar = self.api.get_calendar(start=today, end=today)
            return len(calendar) > 0 and \
                str(calendar[0].date.date() if hasattr(calendar[0].date, 'date')
                    else calendar[0].date) == today
        except Exception as e:
            print(f"  WARN: could not check market calendar: {e}")
            return False


if __name__ == "__main__":
    import config

    broker = AlpacaBroker(
        api_key=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        base_url=config.ALPACA_BASE_URL,
    )

    print("Market open     :", broker.is_market_open())
    print("AAPL position   :", broker.get_position("AAPL"))

    broker.cancel_all_orders()
    print("execution/alpaca_broker.py: OK")
