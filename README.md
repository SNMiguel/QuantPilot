# QuantPilot — Automated Paper Trading System

An end-to-end automated trading system that combines ensemble ML models, live market data, news sentiment analysis, and risk management to trade AAPL, MSFT, and GOOGL on Alpaca's paper trading platform.

![Tests](https://github.com/SNMiguel/QuantPilot/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## What It Does

Every weekday after market close, QuantPilot automatically:
1. Fetches the latest price bar and news sentiment for each ticker
2. Runs leakage-free feature engineering — stationary indicator transforms computed only on past data, targeting the **next-day return** (never the same-day price, which a model can trivially reconstruct from its own features)
3. Loads the latest promoted ensemble model per ticker from the registry
4. Generates a BUY / SELL / HOLD signal with a directional-agreement confidence score
5. Sizes the position using ATR-based risk management (max 15% portfolio exposure)
6. Submits the order to Alpaca and logs the trade and prediction to the database
7. Snapshots portfolio value and sends a Discord summary

Every Sunday, models are retrained on fresh data. A new model is promoted only if it beats the incumbent **on the same held-out window** — comparing metrics measured on different weeks rewards whichever model was tested during a calmer market.

**Live dashboard:** [quantpilot.streamlit.app](https://quantpilot.streamlit.app)

## Honest Evaluation, By Design

Daily equity returns are mostly noise; any pipeline that reports a near-perfect R² on price levels is measuring leakage, not skill. QuantPilot is built to make that failure mode impossible to hide:

- The target is the next-day return, so a persistence baseline ("predict 0") scores exactly zero skill
- **Directional accuracy** is reported everywhere (0.500 = coin flip)
- Every backtest report includes a **buy-and-hold baseline** on the same window, with commission and slippage applied to the strategy
- The test suite ([tests/](tests/)) contains regression tests for the three leakage bugs found and fixed in this codebase: same-day targets, KFold-into-the-future stacking, and stale-fold base models

## Architecture

```
QuantPilot/
├── config.py                     # All settings loaded from .env
│
├── data/
│   ├── database.py               # PostgreSQL via SQLAlchemy (Neon.tech)
│   ├── alpaca_feed.py            # Live + historical price data (IEX feed)
│   └── news_sentiment.py         # NewsAPI + VADER sentiment scoring
│
├── features/
│   ├── indicators.py             # 18 technical indicators (MA, RSI, MACD, BB...)
│   ├── walk_forward.py           # Leakage-free feature slicing by cutoff date
│   └── sentiment_features.py     # Merges sentiment scores onto feature matrix
│
├── models/
│   ├── ensemble.py               # Ridge meta-learner over OOF base predictions
│   ├── registry.py               # JSON manifest + joblib/keras persistence
│   ├── linear_regression.py      # LR, Random Forest, SVR (scikit-learn)
│   ├── neural_network.py         # LSTM + standard/deep/wide architectures
│   └── model_comparison.py       # Trains and evaluates all models side-by-side
│
├── training/
│   ├── walk_forward_trainer.py   # Expanding-window cross-validation + retrain
│   └── metrics.py                # Sharpe, max drawdown, Calmar, win rate, profit factor
│
├── signals/
│   └── generator.py              # BUY/SELL/HOLD from predicted vs current price
│
├── risk/
│   ├── position_sizer.py         # ATR-based sizing with portfolio exposure cap
│   └── portfolio.py              # Syncs positions with Alpaca, tracks drawdown
│
├── execution/
│   ├── alpaca_broker.py          # Order submission, position queries
│   └── order_manager.py          # Full signal → order pipeline with risk checks
│
├── backtest/
│   ├── engine.py                 # Event-driven backtester (next-day open fill)
│   └── report.py                 # Financial metrics + equity curve chart
│
├── monitoring/
│   ├── dashboard.py              # Streamlit dashboard (Altair, dark theme)
│   └── alerts.py                 # Discord webhook notifications
│
├── jobs/
│   ├── daily_job.py              # Runs the full trade pipeline for all tickers
│   ├── train_job.py              # Retrains; promotes only on same-window wins
│   └── backtest_job.py           # On-demand historical strategy evaluation
│
├── tests/                        # Synthetic-data pytest suite (CI on every push)
│
└── .github/workflows/
    ├── daily_trade.yml           # Cron: Mon–Fri 21:30 UTC (5:30 PM ET)
    ├── weekly_retrain.yml        # Cron: Sunday 02:00 UTC
    └── tests.yml                 # pytest on every push and PR
```

## Models

The system trains one **ensemble model per ticker** using a stacked architecture:

- **Target**: next-day simple return, from stationary features (price/MA ratios, normalized MACD, RSI, Bollinger position, volume ratios, momentum returns, volatility, news sentiment)
- **Base models**: Linear Regression, Random Forest, SVR (scikit-learn), each tuned for return-scale targets
- **Meta-learner**: Ridge regression trained on out-of-fold predictions generated with `TimeSeriesSplit`, so every OOF prediction comes from a model that saw only past data; base models are then refit on the full window
- **Confidence**: fraction of base models agreeing with the ensemble's direction — the 0.60 gate means at least two of three must agree
- **Validation**: 5-fold expanding-window walk-forward evaluating the exact ensemble that gets deployed
- **Registry**: versioned JSON manifest with target metadata; jobs refuse to load a model trained for a different target. A Keras LSTM lives in `models/neural_network.py` as an experimental track outside the production ensemble.

## Dashboard

A dark, chart-first Streamlit app (Altair, no matplotlib PNGs):

| Element | Source |
|---|---|
| Equity / cash / return / drawdown KPIs | Alpaca account API + `portfolio_snapshots` |
| Equity curve with drawdown panel | `portfolio_snapshots` DB table |
| Signals tab (latest prediction per ticker) | `predictions` DB table |
| Trades tab | `trades` DB table |
| Sentiment tab (30-day tone per ticker) | `sentiment` DB table |
| Models tab (versions, targets, dir. accuracy) | `models/saved/registry.json` |

## Local Setup

```bash
git clone https://github.com/SNMiguel/QuantPilot.git
cd QuantPilot

python -m venv venv
source venv/Scripts/activate   # Windows Git Bash

pip install -r requirements.txt

# Copy and fill in your credentials
cp .env.example .env
```

**Required `.env` keys:**

```
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
DB_URL=                   # Neon.tech PostgreSQL connection string
NEWS_API_KEY=             # newsapi.org
DISCORD_WEBHOOK_URL=      # optional — alerts channel
LIVE_TRADING=false        # set true only when ready for real money
```

**Initialize and test:**

```bash
python -m data.database         # Create tables
python -m data.alpaca_feed      # Verify Alpaca connection
python -m jobs.train_job        # Train all 3 ticker models (~5 min)
python -m jobs.daily_job        # Run one full trade cycle
streamlit run monitoring/dashboard.py
```

**Backtest a ticker** (includes buy-and-hold comparison, commission, and slippage):

```bash
python -m jobs.backtest_job --ticker AAPL --start 2024-01-01
```

**Run the test suite** (fully synthetic — no API keys or database needed):

```bash
python -m pytest tests/ -v
```

## GitHub Actions

Three automated workflows — no server required:

| Workflow | Schedule | Job |
|---|---|---|
| `daily_trade.yml` | Mon–Fri 21:30 UTC | `jobs/daily_job.py` (orders queue for next open) |
| `weekly_retrain.yml` | Sunday 02:00 UTC | `jobs/train_job.py` |
| `tests.yml` | every push / PR | `pytest tests/` |

**Required GitHub secrets** (Settings → Secrets → Actions):
`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`, `DB_URL`, `NEWS_API_KEY`, `DISCORD_WEBHOOK_URL`, `LIVE_TRADING`

## Risk Management

- **ATR-based position sizing**: `(portfolio × risk_per_trade) / (ATR × multiplier)`
- **Hard exposure cap**: single position ≤ 15% of portfolio value
- **Confidence gate**: at least two of three base models must agree on direction (`CONFIDENCE_THRESHOLD` = 0.60)
- **Drawdown circuit breaker**: all new trades halt if portfolio drawdown exceeds 10% from peak; resuming requires manual review
- **Position-aware execution**: SELL closes the exact held quantity; BUY never stacks onto an open position (no accidental shorts on a cash account)
- **LIVE_TRADING gate**: must be explicitly set to `true` to submit real orders
- **Cash account**: avoids PDT rule (no margin, no 3-trade-per-week limit)

## Technologies

| Layer | Stack |
|---|---|
| ML | scikit-learn (TensorFlow/Keras for the experimental LSTM track) |
| Data | yfinance, Alpaca Markets API, NewsAPI, VADER |
| Database | PostgreSQL (Neon.tech) via SQLAlchemy |
| Dashboard | Streamlit |
| Automation | GitHub Actions |
| Alerts | Discord webhooks |

## Author

**Miguel Shema Ngabonziza**
- LinkedIn: [linkedin.com/in/migztech](https://linkedin.com/in/migztech)
- GitHub: [github.com/SNMiguel](https://github.com/SNMiguel)
- Portfolio: [migztech.vercel.app](https://migztech.vercel.app)

---

If you found this project useful, consider giving it a star!
