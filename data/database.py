"""
PostgreSQL database interface for Neon.tech.
Handles all read/write operations: price bars, sentiment,
predictions, trades, and portfolio snapshots.
"""
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    String, Float, Integer, BigInteger, Date, DateTime, text
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
import pandas as pd
from datetime import datetime


class Database:
    """SQLAlchemy Core interface to the Neon.tech PostgreSQL database."""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.metadata = MetaData()
        self._define_tables()

    def _define_tables(self):
        self.price_bars_table = Table('price_bars', self.metadata,
            Column('ticker', String(10), primary_key=True),
            Column('date',   Date,       primary_key=True),
            Column('open',   Float),
            Column('high',   Float),
            Column('low',    Float),
            Column('close',  Float),
            Column('volume', BigInteger),
        )

        self.sentiment_table = Table('sentiment', self.metadata,
            Column('ticker', String(10), primary_key=True),
            Column('date',   Date,       primary_key=True),
            Column('score',  Float),
        )

        self.predictions_table = Table('predictions', self.metadata,
            Column('ticker',          String(10), primary_key=True),
            Column('date',            Date,       primary_key=True),
            Column('model_version',   String(64), primary_key=True),
            Column('predicted_price', Float),
            Column('signal',          String(10)),
            Column('confidence',      Float),
        )

        self.trades_table = Table('trades', self.metadata,
            Column('order_id',  String(64), primary_key=True),
            Column('ticker',    String(10)),
            Column('side',      String(10)),
            Column('qty',       Integer),
            Column('price',     Float),
            Column('status',    String(20)),
            Column('timestamp', DateTime),
        )

        self.portfolio_snapshots_table = Table('portfolio_snapshots', self.metadata,
            Column('date',  Date,  primary_key=True),
            Column('value', Float),
        )

    def create_tables(self) -> None:
        """Create all tables if they don't exist. Safe to call repeatedly."""
        self.metadata.create_all(self.engine)
        print("Database tables ready.")

    # ------------------------------------------------------------------
    # Price bars
    # ------------------------------------------------------------------

    def upsert_bars(self, df: pd.DataFrame, ticker: str) -> None:
        """Insert or update OHLCV bars. Expects DatetimeIndex + OHLCV columns."""
        records = [
            {
                'ticker': ticker,
                'date':   idx.date() if hasattr(idx, 'date') else idx,
                'open':   float(row['Open']),
                'high':   float(row['High']),
                'low':    float(row['Low']),
                'close':  float(row['Close']),
                'volume': int(row['Volume']),
            }
            for idx, row in df.iterrows()
        ]
        if not records:
            return
        stmt = pg_insert(self.price_bars_table).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=['ticker', 'date'],
            set_={c: stmt.excluded[c] for c in ('open', 'high', 'low', 'close', 'volume')},
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return OHLCV DataFrame for ticker between start and end (inclusive)."""
        query = text("""
            SELECT date, open AS "Open", high AS "High",
                   low  AS "Low",  close AS "Close", volume AS "Volume"
            FROM price_bars
            WHERE ticker = :ticker
              AND date >= :start
              AND date <= :end
            ORDER BY date ASC
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn,
                             params={'ticker': ticker, 'start': start, 'end': end},
                             parse_dates=['date'],
                             index_col='date')
        return df

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def upsert_sentiment(self, date: str, ticker: str, score: float) -> None:
        """Insert or update daily sentiment score for a ticker."""
        stmt = pg_insert(self.sentiment_table).values(
            ticker=ticker, date=date, score=score
        ).on_conflict_do_update(
            index_elements=['ticker', 'date'],
            set_={'score': score},
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_sentiment(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return sentiment scores DataFrame for ticker between start and end."""
        query = text("""
            SELECT date, score
            FROM sentiment
            WHERE ticker = :ticker
              AND date >= :start
              AND date <= :end
            ORDER BY date ASC
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn,
                             params={'ticker': ticker, 'start': start, 'end': end},
                             parse_dates=['date'],
                             index_col='date')
        return df

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def upsert_prediction(self, date: str, ticker: str, model_version: str,
                          predicted_price: float, signal: str,
                          confidence: float) -> None:
        """Record the daily model prediction and generated signal."""
        stmt = pg_insert(self.predictions_table).values(
            ticker=ticker,
            date=date,
            model_version=model_version,
            predicted_price=predicted_price,
            signal=signal,
            confidence=confidence,
        ).on_conflict_do_update(
            index_elements=['ticker', 'date', 'model_version'],
            set_={'predicted_price': predicted_price,
                  'signal': signal,
                  'confidence': confidence},
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_latest_predictions(self) -> pd.DataFrame:
        """Return the most recent prediction row per ticker."""
        query = text("""
            SELECT DISTINCT ON (ticker)
                   ticker, date, model_version,
                   predicted_price, signal, confidence
            FROM predictions
            ORDER BY ticker, date DESC
        """)
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, parse_dates=['date'])

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def log_trade(self, order_id: str, ticker: str, side: str,
                  qty: int, price: float, status: str) -> None:
        """Log a submitted order. Updates status if order_id already exists."""
        stmt = pg_insert(self.trades_table).values(
            order_id=order_id,
            ticker=ticker,
            side=side,
            qty=qty,
            price=price,
            status=status,
            timestamp=datetime.utcnow(),
        ).on_conflict_do_update(
            index_elements=['order_id'],
            set_={'status': status},
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_trade_log(self) -> pd.DataFrame:
        """Return all trades ordered by timestamp descending."""
        query = text("""
            SELECT order_id, ticker, side, qty, price, status, timestamp
            FROM trades
            ORDER BY timestamp DESC
        """)
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, parse_dates=['timestamp'])

    # ------------------------------------------------------------------
    # Portfolio snapshots
    # ------------------------------------------------------------------

    def snapshot_portfolio(self, date: str, value: float) -> None:
        """Record daily portfolio equity value."""
        stmt = pg_insert(self.portfolio_snapshots_table).values(
            date=date, value=value
        ).on_conflict_do_update(
            index_elements=['date'],
            set_={'value': value},
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_portfolio_history(self) -> pd.DataFrame:
        """Return full portfolio value history ordered by date ascending."""
        query = text("SELECT date, value FROM portfolio_snapshots ORDER BY date ASC")
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, parse_dates=['date'], index_col='date')


if __name__ == "__main__":
    import config
    db = Database(config.DB_URL)
    db.create_tables()
    print("Phase 3 verify: OK")
