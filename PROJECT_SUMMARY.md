# QuantPilot — Project Summary

A concise, accurate snapshot of the system. For build history see
[ROADMAP.md](ROADMAP.md); for working rules see [CLAUDE.md](CLAUDE.md).

## What It Is

An automated paper-trading system that trades AAPL, MSFT, and GOOGL on
Alpaca. Every weekday after the close it fetches prices and news, builds
leakage-free features, predicts each ticker's **next-day return** with a
per-ticker stacked ensemble, gates the signal on a conformal prediction
interval, sizes the position by volatility regime and ATR risk, submits
the order (queued for the next open), logs everything, and posts a Discord
summary with a plain-English rationale. Models retrain weekly and are
promoted only when they beat the incumbent on the same held-out window.

## Pipeline

```
Alpaca prices ─┐
NewsAPI+LLM ───┼─▶ leakage-free features (next-day return target)
               │        │
               │        ▼
               │   stacked ensemble (LR/RF/SVR + Ridge meta, TimeSeriesSplit OOF)
               │        │  + split-conformal intervals
               │        ▼
               │   signal (BUY/SELL/HOLD) ─▶ conformal gate ─▶ regime-scaled
               │                                                ATR sizing ─▶ Alpaca
               ▼                                                     │
        Neon PostgreSQL  ◀── trades / predictions / snapshots ──────┘
               │
               ▼
     Streamlit dashboard  +  Discord narration
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Target = next-day **return** (not same-day price) | A same-day price target lets the model reconstruct the answer from its own features — pure leakage. Returns force a real forecast; a persistence baseline scores zero skill. |
| Stationary features (ratios/returns) | Generalize across price regimes and tickers; no raw price levels leak scale. |
| `TimeSeriesSplit` stacking + full-window refit | OOF predictions never see the future; production base models see all data. |
| Promotion on a **shared held-out window** | Comparing stored RMSE from different weeks rewards whichever model was tested in a calmer market. |
| Conformal interval gate | Trade only when the 80% interval excludes zero — principled "the move is not noise". |
| Volatility-regime sizing | Shrink exposure (~40%) when realized-vol percentile is high. |
| LLM sentiment + narrator, with fallbacks | Real event understanding and human-readable rationale, but never a hard dependency. |
| Position-aware execution | SELL closes the held quantity; BUY never stacks — no accidental shorts on a cash account. |
| Drawdown circuit breaker | Halt new trades above 10% drawdown from peak. |

## Configuration (config.py)

| Parameter | Value |
|---|---|
| WATCHLIST | AAPL, MSFT, GOOGL |
| MAX_POSITION_PCT | 0.15 (per ticker) |
| SIGNAL_THRESHOLD | 0.003 (0.3% predicted return; MSFT 0.006) |
| CONFIDENCE_THRESHOLD | 0.60 (directional agreement) |
| USE_CONFORMAL_GATE / CONFORMAL_COVERAGE | true / 0.80 |
| MAX_DRAWDOWN_HALT | 0.10 |
| SLIPPAGE_BPS | 5.0 (backtest) |
| LLM_MODEL | claude-opus-4-8 (optional; VADER fallback) |
| LIVE_TRADING | false |

## Status & Next Steps

- Code paths, ensemble, conformal, regime, sentiment, and narrator are unit-
  tested (synthetic data) and pass in CI.
- **Before first live run:** run `python -m jobs.train_job` to produce
  `next_return` models (the daily job refuses legacy price-target models),
  then a backtest for honest baseline numbers, then verify one `daily_job`
  against real Alpaca/Neon credentials.
- Keep `LIVE_TRADING=false` through 3+ months of clean paper trading.

## Secrets

`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`, `DB_URL`,
`NEWS_API_KEY`, `DISCORD_WEBHOOK_URL`, `LIVE_TRADING`, and optionally
`ANTHROPIC_API_KEY` (+ `LLM_MODEL`). Same keys as GitHub Actions secrets and
Streamlit Cloud secrets.
