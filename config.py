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

# --- Database (Neon.tech PostgreSQL) ---
DB_URL: str = os.getenv("DB_URL")

# --- Trading parameters ---
# One model is trained and stored per ticker: registry key = f"ensemble_{ticker}"
WATCHLIST: list = ["AAPL", "MSFT", "GOOGL"]

TRAIN_LOOKBACK_DAYS: int    = 1000
MAX_POSITION_PCT: float     = 0.15   # 15% per ticker — caps total tech exposure at 45%
CONFIDENCE_THRESHOLD: float = 0.60   # Minimum ensemble agreement to act on a signal
SIGNAL_THRESHOLD: float     = 0.003  # 0.3% predicted move required to trigger BUY/SELL

# Per-ticker overrides — tickers not listed here fall back to SIGNAL_THRESHOLD
SIGNAL_THRESHOLD_OVERRIDES: dict = {
    'MSFT': 0.006,   # MSFT has higher RMSE; require 0.6% move to reduce noise trades
}

# Circuit breaker: halt all new trades if portfolio drawdown from its
# peak exceeds this fraction. Requires manual review to resume.
MAX_DRAWDOWN_HALT: float = 0.10

# Backtest slippage assumption, in basis points of the fill price.
SLIPPAGE_BPS: float = 5.0

# --- Model registry ---
MODEL_REGISTRY_PATH: str = "models/saved/registry.json"
