"""
Backtest job - run on demand to evaluate historical strategy performance.

Usage:
    python -m jobs.backtest_job
    python -m jobs.backtest_job --ticker MSFT
    python -m jobs.backtest_job --ticker AAPL --start 2024-01-01
"""
import argparse
from datetime import date, timedelta

import config
from data.database import Database
from data.alpaca_feed import AlpacaFeed
from models.registry import ModelRegistry
from signals.generator import SignalGenerator
from risk.position_sizer import PositionSizer
from backtest.engine import BacktestEngine
from backtest.report import generate as generate_report


def run_backtest(ticker: str, start: str, end: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Backtest: {ticker}  {start} to {end}")
    print(f"{'='*55}")

    # --- Data ---
    db   = Database(config.DB_URL)
    feed = AlpacaFeed(config.ALPACA_API_KEY,
                      config.ALPACA_SECRET_KEY,
                      config.ALPACA_BASE_URL)

    print(f"Fetching price data...")
    df           = feed.get_historical_bars(ticker, start, end, db=db)
    sentiment_df = db.get_sentiment(ticker, start, end)

    if df.empty:
        print(f"WARN: No data for {ticker}. Aborting.")
        return {}

    print(f"  {len(df)} rows  |  "
          f"sentiment rows: {len(sentiment_df)}")

    # --- Model ---
    registry = ModelRegistry(db=db)
    model, meta = registry.load_latest(f'ensemble_{ticker}',
                                       require_meta={'target': 'next_return'})
    if model is None:
        print(f"WARN: no next-return model found for {ticker}. "
              f"Run train_job.py first.")
        return {}

    print(f"  Model: ensemble_{ticker} [{meta['version_id']}]  "
          f"RMSE={meta['metrics']['rmse']:.5f}  "
          f"dir_acc={meta['metrics'].get('dir_acc', 0):.3f}")

    # --- Components ---
    threshold  = config.SIGNAL_THRESHOLD_OVERRIDES.get(ticker, config.SIGNAL_THRESHOLD)
    signal_gen = SignalGenerator(threshold=threshold)
    sizer      = PositionSizer()
    engine     = BacktestEngine(commission_per_share=0.01,
                                initial_capital=100_000.0,
                                slippage_bps=config.SLIPPAGE_BPS)

    # --- Run ---
    print(f"\nRunning backtest...")
    equity_curve = engine.run(df, sentiment_df, model, signal_gen, sizer)

    if equity_curve.empty:
        print("WARN: no trades were generated. "
              "Try loosening SIGNAL_THRESHOLD or CONFIDENCE_THRESHOLD.")
        return {}

    # --- Report (price_df enables the buy-and-hold baseline) ---
    metrics = generate_report(equity_curve, ticker=ticker, price_df=df)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run strategy backtest')
    parser.add_argument('--ticker', default='AAPL',
                        help='Ticker symbol (default: AAPL)')
    parser.add_argument('--start',  default=None,
                        help='Start date YYYY-MM-DD '
                             '(default: 1 year ago)')
    parser.add_argument('--end',    default=None,
                        help='End date YYYY-MM-DD (default: today)')
    args = parser.parse_args()

    end   = args.end   or date.today().isoformat()
    start = args.start or (date.today() - timedelta(days=365)).isoformat()

    run_backtest(args.ticker, start, end)
