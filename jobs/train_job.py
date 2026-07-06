"""
Weekly model retrain job.

For each ticker in WATCHLIST:
  1. Pull full price + sentiment history from DB
  2. Run WalkForwardTrainer (5-fold expanding window, next-day return target)
  3. Promote to registry only if the challenger beats the incumbent
     on the same held-out window
  4. Send Discord retrain summary

Usage:
    python -m jobs.train_job
    python -m jobs.train_job --ticker AAPL   # single ticker
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import traceback
from datetime import date, timedelta

import config
from data.database import Database
from data.alpaca_feed import AlpacaFeed
from training.walk_forward_trainer import WalkForwardTrainer
from monitoring.alerts import DiscordAlerter


def train_ticker(ticker: str, db: Database, feed: AlpacaFeed,
                 registry=None) -> dict:
    print(f"\n{'='*55}")
    print(f"  Retraining: {ticker}")
    print(f"{'='*55}")

    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=config.TRAIN_LOOKBACK_DAYS)).isoformat()

    # Pull data
    df           = feed.get_historical_bars(ticker, start, end, db=db)
    sentiment_df = db.get_sentiment(ticker, start, end)

    if df.empty:
        print(f"  WARN: No price data for {ticker} - skipping")
        return {}

    print(f"  {len(df)} price rows  |  {len(sentiment_df)} sentiment rows")

    # Train
    trainer = WalkForwardTrainer(n_splits=5,
                                 retrain_window_days=config.TRAIN_LOOKBACK_DAYS,
                                 db=db)
    metrics = trainer.train(df, sentiment_df, ticker=ticker)

    if not metrics:
        print(f"  WARN: Training returned no metrics for {ticker}")
        return {}

    print(f"  RMSE: {metrics.get('rmse', 0):.5f} (return units)  "
          f"dir_acc: {metrics.get('dir_acc', 0):.3f}  "
          f"R2: {metrics.get('r2', 0):.4f}")
    return metrics


def run(tickers: list[str] | None = None) -> None:
    today = date.today().isoformat()
    print(f"\n{'='*55}")
    print(f"  Train Job - {today}")
    print(f"{'='*55}")

    alerter  = DiscordAlerter(config.DISCORD_WEBHOOK_URL)
    watchlist = tickers or config.WATCHLIST

    try:
        db      = Database(config.DB_URL)
        db.create_tables()
        feed    = AlpacaFeed(config.ALPACA_API_KEY,
                             config.ALPACA_SECRET_KEY,
                             config.ALPACA_BASE_URL)

        all_metrics: dict = {}
        for ticker in watchlist:
            m = train_ticker(ticker, db, feed, registry=None)
            if m:
                all_metrics[ticker] = m

        if all_metrics:
            alerter.send_retrain_summary(all_metrics)
            print("\nDiscord retrain summary sent")

        print(f"\n{'='*55}")
        print("  Train job complete")
        print(f"{'='*55}\n")

    except Exception as exc:
        msg = f"train_job failed: {exc}\n{traceback.format_exc()}"
        print(f"\nERROR: {msg}")
        alerter.send_error(msg)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrain models')
    parser.add_argument('--ticker', default=None,
                        help='Single ticker to retrain (default: all watchlist)')
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else None
    run(tickers)
