# QuantPilot

**An automated trading system rigorous enough to tell you the truth: it has no alpha yet.**

![Tests](https://github.com/SNMiguel/QuantPilot/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Claude](https://img.shields.io/badge/LLM-Claude-8A2BE2)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

QuantPilot trades AAPL, MSFT, and GOOGL on Alpaca paper markets with zero human interaction: GitHub Actions fires the pipeline after every close, models retrain weekly, every decision is explained in plain English on Discord, and a live dashboard shows the results. It is a complete production ML system - data, features, models, risk, execution, monitoring, CI - built around one governing principle: **make it impossible to fool yourself.**

**Live dashboard:** [quantpilot.streamlit.app](https://quantpilot.streamlit.app)

---

## The Honest Numbers

Most "stock prediction" repos show a 0.99 R2 and a beautiful equity curve. Both are almost always leakage. Here is what QuantPilot actually reports on itself across all three tickers, walk-forward validated over 2.5 years, with commission and slippage, and with a full multi-year Claude-scored sentiment history in the features:

| Ticker | Walk-forward dir. acc | Strategy return | Buy-and-hold | Trades |
|---|---|---|---|---|
| AAPL | 0.504 | +1.9% | +84.3% | 2 |
| MSFT | 0.478 | -0.4% | -8.9% | 2 |
| GOOGL | 0.501 | +5.0% | +133.4% | 7 |

Directional accuracy of 0.478 - 0.504 is a coin flip (0.500). R2 on next-day returns is ~0.0 (slightly negative) - daily returns are nearly a random walk, and the system says so. Read the results honestly: buy-and-hold beats the strategy on the two names that rose. MSFT is the only case the strategy "wins", and only because MSFT fell 8.9% while the strategy sat mostly in cash - capital preservation in a downtrend, not predictive skill. **There is no edge here yet, and the system is built to tell you that rather than hide it.**

When those models were retrained on the full three-ticker sentiment history, none was promoted: each challenger's RMSE improvement over the incumbent was noise (under the 2% margin), so the promotion gate correctly kept the existing models. Three chances to promote a worse-or-equal model, three rejections.

This is not a failure of engineering - it is the engineering succeeding at its actual job. The early version of this codebase predicted same-day prices and scored R2 = 0.99; that number was a leakage artifact, and finding and killing it reshaped the entire system. Every design decision below exists to make that class of self-deception structurally impossible:

- **The target is the next-day return**, never a price level. A price-level model can trivially reconstruct its target from its own features and look brilliant. The registry stamps every model with its target kind, and jobs refuse to load a model trained for anything else.
- **Stacking uses `TimeSeriesSplit`**, never KFold - KFold leaks the future into out-of-fold predictions and inflates the meta-learner.
- **A challenger replaces the incumbent model only by beating it on the same held-out window, by a real margin** (2% relative RMSE, no directional regression). This gate exists because a noise-level 0.0001 RMSE "win" once promoted a model that backtested strictly worse - the system caught its own mistake.
- **Trades must clear a conformal gate**: the 80% prediction interval, calibrated from out-of-fold residuals, must exclude zero. Not a vibes-based confidence score - a finite-sample statistical guarantee.
- **Every backtest prints the buy-and-hold baseline** on the same window, with the strategy paying commission and slippage while the baseline doesn't pay a management fee.
- **The test suite contains regression tests for the three leakage bugs found and fixed here**: same-day targets, KFold-into-the-future stacking, and stale-fold base models.

The result: a system that correctly refuses to trade on noise, sits mostly in cash, and reports a coin flip as a coin flip. Ask yourself which repo you'd rather have managing money - this one, or the one with the perfect backtest.

---

## What Runs Every Day

Weekdays at 21:30 UTC (after the NYSE close), a GitHub Actions runner executes the full cycle - no server, no human:

```
prices (Alpaca)          news (NewsAPI)
      |                        |
      v                        v
leakage-free features    Claude reads the headlines
(stationary transforms,  -> structured verdict:
 next-day return target)    direction, score, events
      |                        |
      +-----------+------------+
                  v
     per-ticker stacked ensemble
     (LR + RF + SVR -> Ridge meta-learner)
                  |
                  v
     conformal gate: 80% interval
     must exclude zero, or HOLD
                  |
                  v
     volatility regime check
     (turbulent tape -> 0.4x size)
                  |
                  v
     ATR position sizing, 15% cap,
     drawdown circuit breaker
                  |
                  v
     position-aware execution (Alpaca)
                  |
                  v
     Claude narrates WHY, in plain
     English -> Discord + database
```

Sundays, every model retrains on fresh data and faces the promotion gate. Models persist as blobs in Postgres, so they survive the ephemeral CI runners - the Sunday trainer and the Monday trader are different machines that share one durable registry.

A sample of what lands on Discord each evening, generated from the day's actual numbers:

> All three positions remain on HOLD. AAPL and GOOGL show modest positive expected returns with moderate confidence, but neutral sentiment and turbulent volatility regimes warrant caution. MSFT's predicted return of +0.160% is paired with low confidence of 0.33, making it unsuitable for a buy signal. No risk gates were triggered.

---

## The Intelligence Layer

Four components add judgment on top of the forecaster. All four degrade gracefully - remove the Anthropic key and the system keeps trading on simpler paths, never crashing.

**LLM news sentiment** - [data/llm_sentiment.py](data/llm_sentiment.py)
VADER scores words in isolation: "Apple crushes earnings estimates" reads as *negative* because "crushes" is a violent word. Claude reads the headline the way a trader does and returns a structured verdict - direction, score in [-1, 1], confidence, and the concrete events it keyed on ("Q3 earnings beat", "antitrust ruling"). Same output contract as the VADER path, so it drops into the existing feature and DB column. Falls back to VADER without the key.

**Conformal prediction intervals** - [models/ensemble.py](models/ensemble.py)
Split-conformal calibration from out-of-fold residuals wraps every point prediction in an interval with finite-sample coverage (empirically verified in the test suite). The trade gate follows: act only when the 80% interval excludes zero. This replaced a heuristic confidence threshold with a statistical statement about whether the predicted move is distinguishable from noise.

**Volatility-regime sizing** - [features/regime.py](features/regime.py)
Realized-vol percentile classifies the tape as calm / normal / elevated / turbulent and scales position size to 0.4x in turbulent markets. Deliberately a transparent rolling percentile rather than a hidden-state model: it needs no training, cannot silently break, and its output is legible on a dashboard. Applied identically in live trading and the backtester.

**Trade narrator** - [monitoring/narrator.py](monitoring/narrator.py)
Claude receives only the day's computed values - predicted return, confidence, sentiment events, regime, risk-gate outcomes - and writes a few grounded sentences explaining why the system did what it did. It is instructed not to invent market commentary; it explains decisions, it does not editorialize. Falls back to a templated summary.

Try the layer standalone with nothing but an Anthropic key - no broker, no database:

```bash
python demo_local.py
```

---

## Architecture

```
QuantPilot/
├── config.py                     # All settings from .env; safe defaults
│
├── data/
│   ├── alpaca_feed.py            # Live + historical prices, DB-cached
│   ├── news_sentiment.py         # NewsAPI + VADER (fallback path)
│   ├── llm_sentiment.py          # Claude structured event extraction
│   └── database.py               # Neon Postgres via SQLAlchemy Core
│
├── features/
│   ├── indicators.py             # 18 technical indicators
│   ├── walk_forward.py           # Leakage-free features; next-day return target
│   ├── sentiment_features.py     # Sentiment column merge
│   └── regime.py                 # Vol-percentile regime -> size multiplier
│
├── models/
│   ├── ensemble.py               # Stacked LR/RF/SVR + conformal intervals
│   ├── registry.py               # Versioned registry; Postgres blob persistence
│   └── neural_network.py         # Experimental Keras LSTM (not in production)
│
├── training/
│   └── walk_forward_trainer.py   # Expanding-window validation + gated promotion
│
├── signals/  risk/  execution/   # Signal gen, ATR sizing + caps, position-aware orders
│
├── backtest/
│   ├── engine.py                 # Next-day-open fills, commission, slippage, stops
│   └── report.py                 # Metrics + buy-and-hold baseline, always
│
├── monitoring/
│   ├── dashboard.py              # Streamlit + Altair, dark theme
│   ├── narrator.py               # Claude "why we traded" -> Discord
│   └── alerts.py                 # Discord webhooks
│
├── jobs/
│   ├── daily_job.py              # Full trade cycle (--dry-run supported)
│   ├── train_job.py              # Weekly retrain + promotion gate
│   ├── backtest_job.py           # On-demand evaluation
│   └── backfill_sentiment.py     # Historical sentiment via Alpaca news + Claude
│
├── tests/                        # 58 tests, fully synthetic - no keys, DB, or TF
│
└── .github/workflows/
    ├── daily_trade.yml           # Mon-Fri 21:30 UTC
    ├── weekly_retrain.yml        # Sunday 02:00 UTC
    └── tests.yml                 # Every push and PR
```

Engineering details worth noticing:

- **Durable model registry.** GitHub Actions runners are ephemeral; a model trained Sunday would vanish before Monday's job. The registry writes model blobs to Postgres (metrics and metadata in JSONB) and treats the DB as the source of truth, with local files as a cache. Tests run without any database - the DB layer is strictly opt-in.
- **Position-aware execution.** SELL closes exactly the held quantity; BUY never stacks onto an open position. No accidental shorts on a cash account.
- **`--dry-run` mode.** `python -m jobs.daily_job --dry-run` exercises every step - live data, LLM sentiment, model load, conformal gate, regime, sizing - and reports what it *would* do, without submitting orders or posting to Discord.
- **Graceful degradation everywhere.** No Anthropic SDK, no NLTK, no TensorFlow, no API keys - each absence downgrades a feature instead of crashing a job.
- **Sentiment backfill.** NewsAPI's free tier only reaches back 30 days; [jobs/backfill_sentiment.py](jobs/backfill_sentiment.py) pages Alpaca's historical news archive and scores years of headlines through the same Claude pipeline, so backtests see a populated sentiment feature.

---

## Risk Management

| Control | Rule |
|---|---|
| Position sizing | ATR-based: `(portfolio x risk_per_trade) / (ATR x multiplier)` |
| Exposure cap | Single position <= 15% of portfolio |
| Regime scaling | Turbulent volatility -> 0.4x size, elevated -> 0.65x |
| Conformal gate | No trade unless the 80% interval excludes zero |
| Confidence gate | At least 2 of 3 base models must agree on direction |
| Circuit breaker | All trading halts at 10% drawdown; manual review to resume |
| Live-trading gate | `LIVE_TRADING=true` required to touch real money; default false |
| Account type | Cash account - no margin, no PDT exposure |

---

## Run It Yourself

```bash
git clone https://github.com/SNMiguel/QuantPilot.git
cd QuantPilot

python -m venv venv
source venv/Scripts/activate   # Windows Git Bash

pip install -r requirements.txt
cp .env.example .env           # then fill in your keys
```

**`.env` keys:**

```
ALPACA_API_KEY=               # alpaca.markets (paper account)
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
DB_URL=                       # Neon.tech Postgres connection string
NEWS_API_KEY=                 # newsapi.org
DISCORD_WEBHOOK_URL=          # optional - daily summaries
ANTHROPIC_API_KEY=            # optional - Claude sentiment + narration
LLM_MODEL=claude-haiku-4-5    # optional - cheap and fast for daily use
LIVE_TRADING=false            # the only switch that touches real money
```

**Then:**

```bash
python -m pytest tests/ -v                # 58 tests, no credentials needed
python demo_local.py                      # LLM layer demo (Anthropic key only)
python -m jobs.train_job                  # train all 3 ticker models
python -m jobs.backtest_job --ticker AAPL --start 2024-01-01
python -m jobs.daily_job --dry-run        # full cycle, no orders placed
streamlit run monitoring/dashboard.py     # local dashboard
```

**To automate:** add the `.env` keys above as GitHub Actions secrets (Settings -> Secrets and variables -> Actions). The scheduled workflows take it from there.

---

## Technologies

| Layer | Stack |
|---|---|
| ML | scikit-learn stacked ensemble; split-conformal intervals |
| LLM | Anthropic Claude (structured outputs) with deterministic fallbacks |
| Data | Alpaca Markets API, NewsAPI, VADER |
| Database | PostgreSQL (Neon.tech) via SQLAlchemy Core |
| Dashboard | Streamlit + Altair |
| Automation | GitHub Actions (cron; zero servers) |
| Alerts | Discord webhooks |

---

## Roadmap

The infrastructure is finished; the open problem is the honest one - finding edge:

- Features with plausible predictive content at the daily horizon: cross-asset signals, options-implied volatility, earnings-calendar proximity
- Retrain and evaluate against the now-backfilled multi-year LLM sentiment history
- Longer prediction horizons (weekly), where signal-to-noise is friendlier
- Gaussian-HMM regime model behind the existing `detect_regime` interface

## Author

**Miguel Shema Ngabonziza**
- LinkedIn: [linkedin.com/in/migztech](https://linkedin.com/in/migztech)
- GitHub: [github.com/SNMiguel](https://github.com/SNMiguel)
- Portfolio: [migztech.vercel.app](https://migztech.vercel.app)

---

If this project's approach to honest ML evaluation was useful to you, consider giving it a star.
