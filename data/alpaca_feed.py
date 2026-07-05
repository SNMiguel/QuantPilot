"""
Alpaca Markets data feed.
Fetches historical and live OHLCV bars, account info, and market status.
Caches all historical data to the database to avoid redundant API calls.
Falls back to yfinance if Alpaca is unavailable.
"""
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi


class AlpacaFeed:
    """Wrapper around the Alpaca REST API for market data and account info."""

    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2',
        )

    # ------------------------------------------------------------------
    # Historical bars
    # ------------------------------------------------------------------

    def get_historical_bars(self, ticker: str, start: str, end: str,
                            timeframe: str = '1Day',
                            db=None) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a ticker.

        Checks the database cache first. Only fetches from Alpaca the
        date range not already stored. Falls back to yfinance on failure.

        Args:
            ticker:    Stock ticker symbol e.g. 'AAPL'
            start:     Start date string 'YYYY-MM-DD'
            end:       End date string   'YYYY-MM-DD'
            timeframe: Alpaca timeframe string (default '1Day')
            db:        Optional Database instance for caching.

        Returns:
            OHLCV DataFrame with DatetimeIndex, columns:
            Open, High, Low, Close, Volume
        """
        # Check cache
        if db is not None:
            cached = db.get_bars(ticker, start, end)
            if not cached.empty:
                # If cache covers the full range, return it
                cache_start = cached.index[0].strftime('%Y-%m-%d')
                cache_end   = cached.index[-1].strftime('%Y-%m-%d')
                if cache_start <= start and cache_end >= end:
                    return cached

        try:
            bars = self.api.get_bars(
                ticker,
                timeframe,
                start=start,
                end=end,
                adjustment='raw',
                feed='iex',   # free paper accounts use IEX, not SIP
            ).df

            if bars.empty:
                raise ValueError(f"Alpaca returned no data for {ticker}")

            # Alpaca returns a MultiIndex or timezone-aware index — normalise it
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(ticker, level=0)
            bars.index = pd.to_datetime(bars.index).tz_convert(None).normalize()
            bars.index.name = 'date'

            df = bars[['open', 'high', 'low', 'close', 'volume']].copy()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Cache to database
            if db is not None:
                db.upsert_bars(df, ticker)

            return df

        except Exception as e:
            print(f"WARN: Alpaca fetch failed for {ticker}: {e}")
            print("  Falling back to yfinance...")
            return self._yfinance_fallback(ticker, start, end, db)

    def _yfinance_fallback(self, ticker: str, start: str, end: str,
                           db=None) -> pd.DataFrame:
        """Fetch via yfinance when Alpaca is unavailable."""
        import yfinance as yf
        import warnings, logging
        warnings.filterwarnings('ignore')
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)

        raw = yf.download(ticker, start=start, end=end,
                          progress=False, auto_adjust=True)

        if raw.empty:
            print(f"  WARN: yfinance returned no data for {ticker}")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Flatten MultiIndex columns (newer yfinance versions)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Select OHLCV columns — handle any capitalisation
        col_map = {c.lower(): c for c in raw.columns}
        selected = [col_map[c] for c in ('open', 'high', 'low', 'close', 'volume')
                    if c in col_map]
        df = raw[selected].copy()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Normalise index: strip timezone, floor to date
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        df.index = idx.normalize()
        df.index.name = 'date'

        print(f"  yfinance fetched {len(df)} rows for {ticker}")

        if db is not None and not df.empty:
            db.upsert_bars(df, ticker)

        return df

    # ------------------------------------------------------------------
    # Latest bar
    # ------------------------------------------------------------------

    def get_latest_bar(self, ticker: str) -> dict:
        """
        Fetch the most recent OHLCV bar for a ticker.

        Returns:
            dict with keys: open, high, low, close, volume, date
        """
        bars = self.api.get_bars(
            ticker,
            '1Day',
            limit=1,
            adjustment='raw',
            feed='iex',
        ).df

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(ticker, level=0)

        row = bars.iloc[-1]
        return {
            'open':   float(row['open']),
            'high':   float(row['high']),
            'low':    float(row['low']),
            'close':  float(row['close']),
            'volume': int(row['volume']),
            'date':   bars.index[-1].strftime('%Y-%m-%d'),
        }

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """
        Return current account summary.

        Returns:
            dict with keys: equity, cash, portfolio_value
        """
        account = self.api.get_account()
        return {
            'equity':          float(account.equity),
            'cash':            float(account.cash),
            'portfolio_value': float(account.portfolio_value),
        }

    def is_market_open(self) -> bool:
        """Return True if the US market is currently open."""
        clock = self.api.get_clock()
        return clock.is_open


if __name__ == "__main__":
    import config

    feed = AlpacaFeed(
        api_key=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        base_url=config.ALPACA_BASE_URL,
    )

    print("Market open:", feed.is_market_open())
    print("Account:", feed.get_account())

    df = feed.get_historical_bars("AAPL", "2024-01-01", "2024-01-31")
    print(f"Bars fetched: {len(df)} rows")
    print(df.tail(3))
    print("data/alpaca_feed.py: OK")
