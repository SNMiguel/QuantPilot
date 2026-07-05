# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Set up environment
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# Run the test suite (fully synthetic — no API keys, DB, or tensorflow needed)
python -m pytest tests/ -v

# Jobs (require .env with Alpaca/Neon/NewsAPI credentials)
python -m jobs.train_job                 # weekly retrain, all tickers
python -m jobs.train_job --ticker AAPL   # single ticker
python -m jobs.daily_job                 # full daily trade cycle
python -m jobs.backtest_job --ticker AAPL --start 2024-01-01

# Dashboard
streamlit run monitoring/dashboard.py

# Legacy demo (original 80/20 price-prediction comparison; not the production path)
python main.py
```

Most modules also have an `if __name__ == "__main__":` block for isolated smoke-testing.

## Architecture

Production pipeline (jobs/ orchestrates everything):

1. **Data** — `data/alpaca_feed.py` (prices, DB-cached), `data/news_sentiment.py`
   (NewsAPI + VADER), `data/database.py` (Neon PostgreSQL via SQLAlchemy Core).
2. **Features** — `features/walk_forward.py` is the single source of truth:
   - `get_features_at(df, cutoff)` returns `(X, y, dates)` for TRAINING where
     **y is the next-day simple return** and the last row is dropped.
   - `get_latest_features(df, cutoff)` returns the feature DataFrame for
     PREDICTION, keeping the cutoff row.
   - Features are stationary transforms (ratios/returns), listed in
     `FEATURE_COLUMNS`; `features/sentiment_features.py` appends one trailing
     `sentiment` column.
3. **Models** — `models/ensemble.py`: LR/RF/SVR base models (see
   `make_base_models()`; SVR epsilon is tuned for return scale) stacked with a
   Ridge meta-learner fit on `TimeSeriesSplit` out-of-fold predictions, then
   base models are refit on the full window. Confidence = fraction of base
   models agreeing with the ensemble's direction. The Keras LSTM in
   `models/neural_network.py` is experimental and NOT in the production ensemble.
4. **Training** — `training/walk_forward_trainer.py` validates the ensemble
   itself across expanding-window folds, then retrains and promotes to
   `models/registry.py` only if the challenger beats the incumbent **on the
   same held-out window**. Every save is stamped `meta={'target': 'next_return',
   'n_features': ...}`.
5. **Signals/risk/execution** — `signals/generator.py`
   (`generate_from_return()` is the native entry point), `risk/position_sizer.py`
   (ATR sizing + 15% cap), `risk/portfolio.py`, `execution/order_manager.py`
   (position-aware: SELL closes the held qty, BUY never stacks),
   `execution/alpaca_broker.py`.
6. **Backtest** — `backtest/engine.py` (next-day-open fills, commission,
   slippage, stop-loss) and `backtest/report.py` (metrics + buy-and-hold
   baseline when given `price_df`).

## Critical invariants (tests enforce these)

- Model outputs are **next-day returns**, never prices. Jobs must load models
  via `registry.load_latest(prefix, require_meta={'target': 'next_return'})` —
  treating a legacy price-level model's output as a return would be catastrophic.
- Feature/target alignment: `y[t] == close[t+1]/close[t] - 1`. Never let a
  same-day price level into the feature matrix (that was the original leakage bug).
- Ensemble stacking must use `TimeSeriesSplit` (KFold leaks the future into
  OOF predictions) and must refit base models on the full window afterwards.
- The daily job checks `broker.market_traded_today()` (calendar), NOT
  `is_market_open()` (clock) — it runs after the close.
- No emojis in code, output, or docs (owner preference).
- `models/__init__.py` must keep tensorflow imports optional; tests and the
  dashboard run without it.

## Style

- Aligned assignment blocks and section-divider comments follow the existing
  files; match them.
- Metrics vocabulary: RMSE/MAE in return units, plus `dir_acc` (directional
  accuracy, 0.5 = coin flip). Don't report MAPE on returns (division by ~0).
