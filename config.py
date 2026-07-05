"""
Central configuration — reads from .env via python-dotenv.
Import this module anywhere credentials or trading parameters are needed.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Alpaca Markets ---
ALPACA_API_KEY: str    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL: str   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Safety gate: must be explicitly set to "true" in .env to submit live orders.
# Leave as false until 3+ months of clean paper trading have been completed.
LIVE_TRADING: bool = os.getenv("LIVE_TRADING", "false").lower() == "true"

# --- Third-party APIs ---
NEWS_API_KEY: str        = os.getenv("NEWS_API_KEY")
DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL")

# --- Anthropic (LLM news sentiment + trade narration) ---
# Optional. If ANTHROPIC_API_KEY is unset, the system falls back to VADER
# sentiment and templated Discord summaries — no crash, just less insight.
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
# Model for sentiment analysis and trade narration. Defaults to Anthropic's
# most capable Opus tier. For this high-volume daily classification you can
# drop to "claude-haiku-4-5" for lower cost — that is a cost/quality call
# left to you; set LLM_MODEL in .env to override.
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-opus-4-8")

# --- Database (Neon.tech PostgreSQL) ---
DB_URL: str = os.getenv("DB_URL")

# --- Trading parameters ---
# One model is trained and stored per ticker: registry key = f"ensemble_{ticker}"
WATCHLIST: list = ["AAPL", "MSFT", "GOOGL"]

TRAIN_LOOKBACK_DAYS: int    = 1000
MAX_POSITION_PCT: float     = 0.15   # 15% per ticker — caps total tech exposure at 45%
CONFIDENCE_THRESHOLD: float = 0.60   # Minimum ensemble agreement to act on a signal
SIGNAL_THRESHOLD: float     = 0.003  # 0.3% predicted move required to trigger BUY/SELL

# Per-ticker overrides — tickers not listed here fall back to SIGNAL_THRESHOLD.
# These are minimum predicted next-day RETURNS to act on. A higher bar means
# fewer, higher-conviction trades on a noisier ticker.
SIGNAL_THRESHOLD_OVERRIDES: dict = {
    'MSFT': 0.006,   # require a 0.6% predicted move on MSFT to cut noise trades
}

# Circuit breaker: halt all new trades if portfolio drawdown from its
# peak exceeds this fraction. Requires manual review to resume.
MAX_DRAWDOWN_HALT: float = 0.10

# Conformal gate: when True, a BUY/SELL is only acted on if the model's
# split-conformal prediction interval at CONFORMAL_COVERAGE excludes zero
# (i.e. the predicted move is statistically distinguishable from noise).
# This is a principled alternative to the directional-agreement confidence.
USE_CONFORMAL_GATE: bool = os.getenv("USE_CONFORMAL_GATE", "true").lower() == "true"
CONFORMAL_COVERAGE: float = 0.80

# Backtest slippage assumption, in basis points of the fill price.
SLIPPAGE_BPS: float = 5.0

# --- Model registry ---
MODEL_REGISTRY_PATH: str = "models/saved/registry.json"
