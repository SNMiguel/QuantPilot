# Production Trading System — Roadmap

> **HISTORICAL BUILD LOG — superseded as the source of truth.**
> This file records how the system was built out from the original demo
> (Phases 1–13). The architecture has since moved on and this document is
> **no longer authoritative**. For how the system actually works today, read
> [CLAUDE.md](CLAUDE.md) and [README.md](README.md). Notably, since this log
> was written: the model target changed from same-day price to **next-day
> return**, promotion is decided on a **shared held-out window** (not stored
> RMSE), and the pipeline gained **LLM news sentiment**, **conformal
> prediction intervals**, **volatility-regime position sizing**, an
> **LLM trade narrator**, and a **pytest suite + CI**. Descriptions below
> that mention a same-day price target or RMSE-based promotion are stale.

Tracks every planned change to evolve this project from an ML demo into a
paper trading system. Items are grouped by phase. Status is updated as work
progresses.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## How to Use This File

This is a **living specification**. Any AI assistant (Claude, Cursor, Copilot)
should read the entire file before touching any code.

- When starting a task: change `[ ]` to `[~]`
- When finished: change `[~]` to `[x]`
- After any non-obvious implementation decision: add a one-line note indented
  below the checkbox explaining what was decided and why
- Append a dated entry to **Handoff Notes** (bottom of file) whenever you
  finish a phase or make a significant architectural choice

The **Current Codebase Inventory** section below describes everything that
already exists. Never recreate something listed there — extend it instead.

---

## Current Codebase Inventory

Everything listed here exists in the repo right now. Check here before writing
any new code to avoid duplication.

### `utils/data_loader.py` — `StockDataLoader`
```
__init__(ticker="AAPL", start_date="2020-01-01", end_date=None)
download_data(use_sample_if_fails=True) -> DataFrame
  # Downloads OHLCV from yfinance; stores in self.data and self.raw_data
  # Falls back to utils/sample_data.generate_sample_aapl_data() on failure
add_technical_indicators() -> DataFrame
  # Adds 18 columns IN-PLACE to self.data:
  # MA_5, MA_10, MA_20, MA_50, EMA_12, EMA_26, MACD, Signal_Line,
  # RSI, BB_Middle, BB_Upper, BB_Lower, Volume_MA_5, Volume_MA_20,
  # Momentum_5, Momentum_10, Daily_Return, Volatility
prepare_features(target_column='Close', drop_na=True) -> (X, y, dates)
  # Drops Open/High/Low/Close/Volume/Adj Close from X
  # Stores feature names in self.feature_names
  # Returns numpy arrays X, y and DatetimeIndex dates
get_basic_features() -> (X, y, dates)
  # Legacy: day-number index as only feature. Not used in main pipeline.
```

### `models/linear_regression.py` — `TraditionalModels`
```
__init__()
  # self.models = {}   (populated by train_* methods)
  # self.scaler = StandardScaler()
train_all(X_train, y_train) -> None
  # Calls train_linear_regression, train_random_forest, train_svr in order
  # IMPORTANT: train_all does NOT scale — ModelComparison passes raw X
train_linear_regression(X_train, y_train) -> model
train_random_forest(X_train, y_train, n_estimators=100) -> model
  # max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
train_svr(X_train, y_train) -> model
  # kernel='rbf', C=100, gamma='scale', epsilon=0.1
predict(model_name: str, X) -> ndarray
predict_all(X) -> dict[str, ndarray]
save_models(directory='models/saved') -> None   # joblib
load_models(directory='models/saved') -> None   # joblib
# NOTE: prepare_data() exists on this class but is NOT called by
# ModelComparison — the 80/20 split happens in main.py
```

### `models/neural_network.py` — `NeuralNetworkModel`
```
__init__(input_dim: int)
  # self.scaler = StandardScaler()  ← NN owns its OWN scaler
  # self.model = None
  # self.history = None
build_model(architecture='standard') -> keras.Model
  # 'standard': Dense(64)→Dropout(0.2)→Dense(32)→Dropout(0.2)→Dense(16)→Dense(1)
  # 'deep':     adds BatchNormalization layers, Dropout(0.3)
  # 'wide':     Dense(256)→Dropout(0.3)→Dense(128)→Dropout(0.2)→Dense(1)
  # Compiled: Adam(lr=0.001), loss='mse', metrics=['mae','mse']
train(X_train, y_train, epochs=100, batch_size=32,
      validation_split=0.1, verbose=1) -> history
  # EarlyStopping(patience=15), ReduceLROnPlateau(patience=10)
predict(X) -> ndarray   # model.predict().flatten()
evaluate(X_test, y_test) -> dict
save_model(filepath='models/saved/neural_network.keras') -> None
load_model(filepath='models/saved/neural_network.keras') -> keras.Model
get_model_summary() -> None
```

### `models/model_comparison.py` — `ModelComparison`
```
__init__()
  # self.traditional_models = TraditionalModels()
  # self.nn_model = None
  # self.evaluator = ModelEvaluator()
  # self.results = {}
train_all_models(X_train, y_train, input_dim) -> None
  # Calls traditional_models.train_all(X_train, y_train)  ← raw X, no scaling
  # Builds NeuralNetworkModel, fits nn_model.scaler on X_train, then trains
evaluate_all_models(X_test, y_test, dates_test) -> None
  # Populates self.results:
  # { model_name: {'predictions': ndarray, 'metrics': dict} }
  # Traditional models: raw X_test
  # Neural network: nn_model.scaler.transform(X_test)
get_best_model() -> (str, dict)   # by lowest RMSE
print_comparison() -> None
plot_all_predictions(dates_test, y_test, save_dir='results') -> None
```

### `utils/evaluation.py` — `ModelEvaluator`
```
__init__()
  # self.results = {}  keyed by model_name
calculate_metrics(y_true, y_pred, model_name) -> dict
  # Returns and stores: MAE, MSE, RMSE, R², MAPE
print_metrics(metrics, model_name) -> None
plot_predictions(dates, y_true, y_pred, model_name, save_path=None) -> None
plot_residuals(y_true, y_pred, model_name, save_path=None) -> None
  # Two subplots: scatter (residuals vs predicted) + histogram
compare_models(save_path=None) -> None
  # 2×2 grid: MAE, RMSE, R², MAPE bar charts for all stored models
print_comparison_table() -> None
```

### `utils/sample_data.py`
```
generate_sample_aapl_data(start_date="2020-01-01", end_date="2024-01-01")
  -> DataFrame  # columns: Open, High, Low, Close, Volume
                # index:   business-day DatetimeIndex
                # prices:  linear trend $75→$185 + noise + seasonal sine
```

### `models/__init__.py`
```python
from .linear_regression import TraditionalModels
from .neural_network import NeuralNetworkModel
from .model_comparison import ModelComparison
```

### `main.py`
Linear orchestration script — no classes. Runs the full 6-step pipeline:
load → indicators → split (80/20 chronological) → train → evaluate → visualize.
Entry point for the original demo. Not modified in this roadmap.

---

## Phase 1 — Foundation (Prerequisites)

External accounts to create before any code is written.

- [x] Create Alpaca Markets account → generate paper trading API keys
- [x] Create NewsAPI.org account → generate free API key
- [x] Create Discord server → create a webhook URL for alerts
- [x] Add all keys to `.env` for local runs (GitHub Actions secrets → Phase 13)
- [x] Decide on database host → Neon.tech PostgreSQL

---

## Phase 2 — Repository Restructure

No logic changes — just creating package directories.

- [x] Create `data/` package (`__init__.py`)
- [x] Create `features/` package (`__init__.py`)
- [x] Create `training/` package (`__init__.py`)
- [x] Create `signals/` package (`__init__.py`)
- [x] Create `risk/` package (`__init__.py`)
- [x] Create `execution/` package (`__init__.py`)
- [x] Create `backtest/` package (`__init__.py`)
- [x] Create `monitoring/` package (`__init__.py`)
- [x] Create `jobs/` package (`__init__.py`)
- [x] Move indicator logic out of `utils/data_loader.py` → `features/indicators.py`
- [x] Update all imports after the move
- [x] Add `.env.example` with all required variable names (no values)

---

## Phase 3 — Config & Database

### `config.py` (NEW — project root)
```python
# No class. Module-level constants via python-dotenv.
from dotenv import load_dotenv
load_dotenv()

ALPACA_API_KEY: str         = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str      = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL: str        = os.getenv("ALPACA_BASE_URL",
                                "https://paper-api.alpaca.markets")
# LIVE_TRADING: must be explicitly set to "true" to submit real orders.
# Defaults to False. daily_job.py prints a confirmation prompt before
# ANY live order if this is True. Never set this automatically.
LIVE_TRADING: bool          = os.getenv("LIVE_TRADING", "false").lower() == "true"
NEWS_API_KEY: str           = os.getenv("NEWS_API_KEY")
DISCORD_WEBHOOK_URL: str    = os.getenv("DISCORD_WEBHOOK_URL")
DB_URL: str                 = os.getenv("DB_URL")  # Neon.tech postgres:// connection string
# Multi-ticker watchlist (Q3 decision: AAPL, MSFT, GOOGL)
WATCHLIST: list[str]        = ["AAPL", "MSFT", "GOOGL"]
TRAIN_LOOKBACK_DAYS: int    = 1000
# 15% per ticker (not 20%) — 3 correlated tech stocks, cap total exposure at 45%
MAX_POSITION_PCT: float     = 0.15
CONFIDENCE_THRESHOLD: float = 0.60
SIGNAL_THRESHOLD: float     = 0.01   # 1% predicted move to trigger signal
MODEL_REGISTRY_PATH: str    = "models/saved/registry.json"
```

Tasks:
- [x] **`config.py`** — implement as above
- [x] **`.env.example`** — list all variable names with placeholder values

### `data/database.py` (NEW) — class `Database`
```python
__init__(db_path: str)
get_engine() -> sqlalchemy.Engine
create_tables() -> None
  # Creates tables if not exist (idempotent). Call at startup of every job.
  # Tables: price_bars, features, predictions, trades,
  #         portfolio_snapshots, sentiment
upsert_bars(df: DataFrame, ticker: str) -> None
  # INSERT OR REPLACE into price_bars keyed on (ticker, date)
get_bars(ticker: str, start: str, end: str) -> DataFrame
  # Returns OHLCV DataFrame for date range
upsert_sentiment(date: str, ticker: str, score: float) -> None
get_sentiment(ticker: str, start: str, end: str) -> DataFrame
log_trade(order_id, ticker, side, qty, price, status) -> None
snapshot_portfolio(date: str, value: float) -> None
get_portfolio_history() -> DataFrame   # All portfolio_snapshots rows
get_trade_log() -> DataFrame           # All trades rows
```

Tasks:
- [x] **`data/database.py`** — implement `Database` class

### Verify Phase 3
```bash
python -c "
from data.database import Database
db = Database('data/trading.db')
db.create_tables()
print('Phase 3 OK')
"
```

---

## Phase 4 — Data Layer

### `data/alpaca_feed.py` (NEW) — class `AlpacaFeed`
```python
__init__(api_key: str, secret_key: str, base_url: str)
get_historical_bars(ticker, start, end, timeframe='1Day') -> DataFrame
  # 1. Check DB for cached data (database.get_bars)
  # 2. Fetch missing range from Alpaca REST API
  # 3. Store new data via database.upsert_bars()
  # 4. Falls back to StockDataLoader.download_data() on Alpaca failure
  # Returns OHLCV DataFrame with DatetimeIndex
get_latest_bar(ticker: str) -> dict
  # Returns {'open', 'high', 'low', 'close', 'volume', 'date'}
get_account() -> dict
  # Returns {'equity', 'cash', 'portfolio_value'} from Alpaca account endpoint
is_market_open() -> bool
  # Calls Alpaca clock endpoint
```

### `data/news_sentiment.py` (NEW) — class `NewsSentiment`
```python
__init__(api_key: str)
fetch_articles(ticker: str, date: str) -> list[dict]
  # GET newsapi.org/v2/everything?q=<ticker>&from=<date>&to=<date>&pageSize=5
  # Returns list of article dicts with 'title' and 'description'
score_articles(articles: list[dict]) -> float
  # nltk.sentiment.vader.SentimentIntensityAnalyzer (local, no API cost)
  # Scores title + description for each article; returns mean compound [-1, 1]
  # Returns 0.0 if articles list is empty
get_daily_score(ticker: str, date: str) -> float
  # Calls fetch_articles → score_articles → database.upsert_sentiment()
  # Returns the score
```

Tasks:
- [x] **`data/alpaca_feed.py`** — implement `AlpacaFeed`
- [x] **`data/news_sentiment.py`** — implement `NewsSentiment`

### Verify Phase 4
```bash
# AlpacaFeed (requires .env with real keys)
python -c "
import config
from data.alpaca_feed import AlpacaFeed
feed = AlpacaFeed(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_BASE_URL)
print('Market open:', feed.is_market_open())
print('Account:', feed.get_account())
"

# NewsSentiment (requires .env with NEWS_API_KEY)
python -c "
import config
from data.news_sentiment import NewsSentiment
ns = NewsSentiment(config.NEWS_API_KEY)
score = ns.get_daily_score('AAPL', '2024-01-15')
print('Sentiment score:', score)
"
```

---

## Phase 5 — Feature Engineering (Correctness Fix)

**Critical**: The current `data_loader.py` computes indicators over the full
dataset before splitting — this leaks future data into training features via
rolling windows. Phase 5 fixes this.

### `features/indicators.py` (NEW — extracted from `StockDataLoader`)
```python
def add_indicators(df: DataFrame) -> DataFrame:
  # REUSE: identical logic to StockDataLoader.add_technical_indicators()
  # Difference: standalone function, does not mutate self.data
  # Input:  DataFrame with Open/High/Low/Close/Volume columns
  # Output: same DataFrame with 18 indicator columns appended
  # Columns added: MA_5, MA_10, MA_20, MA_50, EMA_12, EMA_26, MACD,
  #   Signal_Line, RSI, BB_Middle, BB_Upper, BB_Lower, Volume_MA_5,
  #   Volume_MA_20, Momentum_5, Momentum_10, Daily_Return, Volatility
```

### `features/walk_forward.py` (NEW)
```python
def get_features_at(df: DataFrame, cutoff_date: str
                    ) -> tuple[ndarray, ndarray, Index]:
  # Slices df to rows <= cutoff_date BEFORE computing indicators
  # Calls features.indicators.add_indicators() on the sliced data
  # Drops Open/High/Low/Close/Volume/Adj Close (same as prepare_features)
  # Drops NaN rows
  # Returns (X, y, dates) — same contract as StockDataLoader.prepare_features()
```

### `features/sentiment_features.py` (NEW)
```python
def merge_sentiment(feature_df: DataFrame, sentiment_df: DataFrame
                    ) -> DataFrame:
  # Left-joins sentiment_df onto feature_df by date index
  # Forward-fills NaN sentiment values (covers weekends and holidays)
  # Fills any remaining NaN with 0.0 (neutral)
  # Returns feature_df with one extra 'sentiment' column
```

Tasks:
- [x] **`features/indicators.py`** — extract from `StockDataLoader.add_technical_indicators()`
- [x] **`features/walk_forward.py`** — implement `get_features_at()`
- [x] **`features/sentiment_features.py`** — implement `merge_sentiment()`
- [x] Update `StockDataLoader.add_technical_indicators()` to delegate to
      `features.indicators.add_indicators()` (keeps backward compat)

### Verify Phase 5
```bash
python -c "
from utils.data_loader import StockDataLoader
from features.walk_forward import get_features_at

loader = StockDataLoader('AAPL', '2020-01-01', '2024-01-01')
loader.download_data()
df = loader.raw_data

X_wf, y_wf, d_wf = get_features_at(df, '2022-01-01')
X_orig, y_orig, d_orig = loader.prepare_features()

print('walk_forward shape:', X_wf.shape)   # should have 19 features (18 + sentiment ready)
print('original shape:', X_orig.shape)
print('no future leak: cutoff respected:', d_wf[-1] <= '2022-01-01')
"
```

---

## Phase 6 — Model Improvements

### `models/neural_network.py` — Add LSTM architecture (MODIFY)
```python
# Add to build_model(architecture=...) method:
elif architecture == 'lstm':
    # Requires X reshaped to (samples, sequence_length, features)
    self.model = keras.Sequential([
        layers.LSTM(64, return_sequences=True,
                    input_shape=(self.sequence_length, self.input_dim)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
# Add sequence_length parameter to __init__:
# __init__(self, input_dim: int, sequence_length: int = 10)
```

### `models/registry.py` (NEW) — class `ModelRegistry`
```python
__init__(registry_path: str = config.MODEL_REGISTRY_PATH)
  # Creates models/saved/ directory if not exists
  # Loads registry.json if exists, else starts with empty list
save(model, name: str, metrics: dict, framework: str) -> str
  # framework: 'sklearn' | 'keras'
  # sklearn → joblib.dump to models/saved/<version_id>.joblib
  # keras   → model.save to models/saved/<version_id>.keras
  # Appends to registry.json: {version_id, name, path, metrics, timestamp, framework}
  # Returns version_id (uuid4 string)
load_best(metric: str = 'rmse') -> tuple[model, dict]
  # Finds entry with lowest metrics[metric] in registry.json
  # Loads and returns (model, entry_dict)
load_version(version_id: str) -> model
list_versions() -> list[dict]
  # Returns all entries from registry.json sorted by timestamp desc
```

### `models/ensemble.py` (NEW) — class `EnsembleModel`
```python
__init__(base_models: dict)
  # base_models: {'Linear Regression': fitted_model, 'Random Forest': ...}
  # self.meta_learner = Ridge(alpha=1.0)
fit(X_train: ndarray, y_train: ndarray, n_splits: int = 5) -> None
  # Time-series KFold (no shuffle) to generate out-of-fold predictions
  # Stacks OOF predictions: shape (n_train, n_base_models)
  # Fits self.meta_learner on stacked OOF predictions vs y_train
predict(X: ndarray) -> ndarray
  # Gets predictions from all base_models → stacks → meta_learner.predict()
get_confidence(X: ndarray) -> float
  # Std deviation of base model predictions across models, normalized by mean
  # Lower value = higher agreement = higher confidence
  # Returns scalar confidence in [0, 1]
```

Tasks:
- [x] **`models/neural_network.py`** — add `sequence_length` param and `'lstm'` architecture
- [x] **`models/registry.py`** — implement `ModelRegistry`
- [x] **`models/ensemble.py`** — implement `EnsembleModel`

### Verify Phase 6
```bash
python -c "
import numpy as np
from models.registry import ModelRegistry
from sklearn.linear_model import LinearRegression

reg = ModelRegistry()
m = LinearRegression().fit(np.random.randn(100,5), np.random.randn(100))
vid = reg.save(m, 'test_lr', {'rmse': 5.2, 'r2': 0.91}, 'sklearn')
print('Saved version:', vid)
print('Versions:', reg.list_versions())
loaded, meta = reg.load_best('rmse')
print('Loaded:', type(loaded).__name__, 'metrics:', meta['metrics'])
"
```

---

## Phase 7 — Walk-Forward Training

### `training/metrics.py` (NEW)
```python
def sharpe_ratio(returns: ndarray, risk_free_rate: float = 0.0) -> float:
    # (mean(returns) - risk_free_rate) / std(returns) * sqrt(252)
def max_drawdown(equity_curve: ndarray) -> float:
    # max((peak - trough) / peak) over all peaks
def calmar_ratio(returns: ndarray, equity_curve: ndarray) -> float:
    # annualized_return / abs(max_drawdown)
def win_rate(trade_log: DataFrame) -> float:
    # count(pnl > 0) / count(all trades)
def profit_factor(trade_log: DataFrame) -> float:
    # sum(winning pnl) / abs(sum(losing pnl))
```

### `training/walk_forward_trainer.py` (NEW) — class `WalkForwardTrainer`
```python
__init__(n_splits: int = 5, retrain_window_days: int = 500)
train(df: DataFrame, sentiment_df: DataFrame) -> dict
  # 1. Generate n_splits folds:
  #    Each fold: train window = [fold_start, cutoff], test = [cutoff, cutoff+step]
  #    Uses features.walk_forward.get_features_at(df, cutoff)
  #    Merges sentiment via features.sentiment_features.merge_sentiment()
  # 2. Per fold:
  #    a. Train ModelComparison (reuses existing class)
  #    b. Train EnsembleModel on top of ModelComparison base models
  #    c. Evaluate on fold test set; collect metrics
  # 3. Average metrics across all folds
  # 4. Retrain on most recent retrain_window_days (full final window)
  # 5. Save via ModelRegistry.save() ONLY if RMSE beats current registry best
  # Returns dict of averaged metrics across folds
```

Tasks:
- [x] **`training/metrics.py`** — implement all 5 financial metric functions
- [x] **`training/walk_forward_trainer.py`** — implement `WalkForwardTrainer`

### Verify Phase 7
```bash
python -c "
from data.database import Database
from data.alpaca_feed import AlpacaFeed
from data.news_sentiment import NewsSentiment
from training.walk_forward_trainer import WalkForwardTrainer
import config

db = Database(config.DB_PATH)
feed = AlpacaFeed(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_BASE_URL)
df = feed.get_historical_bars(config.TICKER, '2020-01-01', '2024-01-01')
sentiment_df = db.get_sentiment(config.TICKER, '2020-01-01', '2024-01-01')

trainer = WalkForwardTrainer(n_splits=3)
metrics = trainer.train(df, sentiment_df)
print('Walk-forward metrics:', metrics)
"
```

---

## Phase 8 — Signal & Risk Layer

### `signals/generator.py` (NEW) — class `SignalGenerator`
```python
__init__(threshold: float = 0.01, confidence_threshold: float = 0.60)
generate(current_price: float, predicted_price: float,
         confidence: float) -> dict:
  # delta_pct = (predicted_price - current_price) / current_price
  # signal = 'BUY'  if delta_pct > threshold AND confidence >= confidence_threshold
  #        = 'SELL' if delta_pct < -threshold AND confidence >= confidence_threshold
  #        = 'HOLD' otherwise
  # Returns {
  #   'signal': str,
  #   'confidence': float,
  #   'predicted': float,
  #   'current': float,
  #   'delta_pct': float
  # }
```

### `risk/position_sizer.py` (NEW) — class `PositionSizer`
```python
__init__(risk_per_trade: float = 0.01,
         atr_multiplier: float = 2.0,
         max_position_pct: float = 0.20)
calculate_atr(df: DataFrame, period: int = 14) -> float:
  # Average True Range over last `period` rows of df
  # True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
size(portfolio_value: float, current_price: float, atr: float) -> int:
  # raw_shares = (portfolio_value * risk_per_trade) / (atr * atr_multiplier)
  # cap_shares  = (portfolio_value * max_position_pct) / current_price
  # return int(min(raw_shares, cap_shares))  ← floor, never 0
```

### `risk/portfolio.py` (NEW) — class `Portfolio`
```python
__init__(feed: AlpacaFeed, db: Database)
get_open_positions() -> dict
  # Calls feed (AlpacaFeed) account positions endpoint
  # Returns {ticker: {'qty': int, 'market_value': float}}
get_portfolio_value() -> float
  # feed.get_account()['equity']
is_within_limits(ticker: str, shares: int, price: float) -> bool:
  # new_exposure = shares * price
  # total_equity = get_portfolio_value()
  # return (new_exposure / total_equity) <= config.MAX_POSITION_PCT
snapshot(date: str) -> None:
  # db.snapshot_portfolio(date, get_portfolio_value())
get_max_drawdown() -> float:
  # db.get_portfolio_history() → equity curve → training.metrics.max_drawdown()
```

Tasks:
- [x] **`signals/generator.py`** — implement `SignalGenerator`
- [x] **`risk/position_sizer.py`** — implement `PositionSizer`
- [x] **`risk/portfolio.py`** — implement `Portfolio`

### Verify Phase 8
```bash
python -c "
from signals.generator import SignalGenerator
sg = SignalGenerator(threshold=0.01, confidence_threshold=0.6)
print(sg.generate(150.0, 153.0, 0.75))   # expect BUY
print(sg.generate(150.0, 151.0, 0.75))   # expect HOLD (< 1%)
print(sg.generate(150.0, 153.0, 0.50))   # expect HOLD (low confidence)
print(sg.generate(150.0, 147.0, 0.75))   # expect SELL
"
```

---

## Phase 9 — Execution Layer

### `execution/alpaca_broker.py` (NEW) — class `AlpacaBroker`
```python
__init__(api_key: str, secret_key: str, base_url: str)
submit_order(ticker: str, qty: int, side: str,
             order_type: str = 'market') -> dict:
  # Calls Alpaca POST /v2/orders
  # side: 'buy' | 'sell'
  # Returns Alpaca order object dict
  # Raises on API error (let OrderManager catch and log)
get_position(ticker: str) -> dict | None:
  # GET /v2/positions/<ticker>; returns None if no position
cancel_all_orders() -> None:
  # DELETE /v2/orders
is_market_open() -> bool:
  # GET /v2/clock → clock['is_open']
```

### `execution/order_manager.py` (NEW) — class `OrderManager`
```python
__init__(broker: AlpacaBroker, portfolio: Portfolio,
         sizer: PositionSizer, db: Database,
         alerts: DiscordAlerter)
execute_signal(signal: dict, ticker: str,
               price_df: DataFrame) -> dict | None:
  # 1. Return None immediately if signal['signal'] == 'HOLD'
  # 2. atr = sizer.calculate_atr(price_df)
  # 3. qty = sizer.size(portfolio.get_portfolio_value(),
  #                     signal['current'], atr)
  # 4. Return None if qty == 0
  # 5. portfolio.is_within_limits(ticker, qty, signal['current'])
  #    → Return None if limit breached
  # 6. order = broker.submit_order(ticker, qty, signal['signal'].lower())
  # 7. db.log_trade(order['id'], ticker, signal['signal'],
  #                 qty, signal['current'], 'submitted')
  # 8. alerts.send_order_alert(ticker, signal['signal'], qty, signal['current'])
  # 9. Return order dict
```

Tasks:
- [x] **`execution/alpaca_broker.py`** — implement `AlpacaBroker`
- [x] **`execution/order_manager.py`** — implement `OrderManager`

### Verify Phase 9
```bash
# Run full daily job in dry-run mode (paper trading, no real money)
python jobs/daily_job.py --dry-run
# Expected output:
#   Market open check: ...
#   Latest bar fetched: ...
#   Signal generated: {...}
#   Order result: {...} or "HOLD — no order submitted"
# Check Discord for alert message
# Check data/trading.db trades table for log entry
```

---

## Phase 10 — Backtesting

### `backtest/engine.py` (NEW) — class `BacktestEngine`
```python
__init__(commission_per_share: float = 0.01,
         initial_capital: float = 100_000.0)
run(df: DataFrame, sentiment_df: DataFrame,
    model,   # any object with .predict(X) -> ndarray
    signal_gen: SignalGenerator,
    sizer: PositionSizer) -> DataFrame:
  # Iterates each date t in df chronologically:
  #   X, y, dates = get_features_at(df, t)
  #   X_merged = merge_sentiment(X_as_df, sentiment_df)
  #   predicted = model.predict(X_merged[-1:])
  #   current_price = df.loc[t, 'Close']
  #   confidence = model.get_confidence(X_merged[-1:])  if EnsembleModel
  #                else 0.7  (default for single models)
  #   signal = signal_gen.generate(current_price, predicted[0], confidence)
  #   if signal != HOLD:
  #     qty = sizer.size(portfolio_value, current_price, atr)
  #     fill_price = df.loc[t+1, 'Open']  (next day open)
  #     update cash, position, portfolio_value
  #     apply commission: cash -= qty * commission_per_share
  # Returns DataFrame: columns = [date, portfolio_value, cash, position_qty,
  #                                trade_side, trade_qty, trade_price]
```

### `backtest/report.py` (NEW)
```python
def generate(equity_curve: DataFrame, trade_log: DataFrame) -> dict:
  # Computes returns from equity_curve['portfolio_value']
  # Calls all training.metrics functions
  # Saves results/backtest_equity_curve.png (matplotlib line chart)
  # Prints formatted table: Total Return, Sharpe, Max Drawdown,
  #   Calmar, Win Rate, Profit Factor, # Trades
  # Returns dict of all computed metrics
```

### `jobs/backtest_job.py` (NEW)
```python
# 1. db.get_bars(TICKER, start, end) → df
# 2. db.get_sentiment(TICKER, start, end) → sentiment_df
# 3. model, _ = registry.load_best('rmse')
# 4. engine = BacktestEngine()
# 5. equity_curve = engine.run(df, sentiment_df, model, signal_gen, sizer)
# 6. backtest.report.generate(equity_curve, trade_log=equity_curve)
```

Tasks:
- [x] **`backtest/engine.py`** — implement `BacktestEngine`
- [x] **`backtest/report.py`** — implement `generate()`
- [x] **`jobs/backtest_job.py`** — implement orchestration script

### Verify Phase 10
```bash
python jobs/backtest_job.py
# Expected: printed metrics table + results/backtest_equity_curve.png saved
```

---

## Phase 11 — Monitoring & Alerts

### `monitoring/alerts.py` (NEW) — class `DiscordAlerter`
```python
__init__(webhook_url: str)
  # self.webhook_url = webhook_url
send_order_alert(ticker: str, side: str, qty: int, price: float) -> None:
  # POST {"content": "ORDER: <side> <qty> <ticker> @ $<price>"}
send_daily_summary(signal: dict, portfolio_value: float) -> None:
  # POST formatted embed: signal, predicted price, portfolio value
send_retrain_summary(metrics: dict) -> None:
  # POST: new model RMSE, R², Sharpe vs previous best
send_error(error_message: str) -> None:
  # POST: "ERROR: <message>"
# All methods: requests.post(); catch all exceptions silently (never crash job)
```

### `monitoring/dashboard.py` (NEW) — Streamlit app
```python
# Run: streamlit run monitoring/dashboard.py
# Page sections:
#   1. Portfolio Value — feed.get_account()['equity'] live
#   2. Equity Curve   — db.get_portfolio_history() line chart
#   3. Recent Trades  — db.get_trade_log() dataframe table (last 20 rows)
#   4. Today's Signal — db latest predictions row
#   5. Model Versions — registry.list_versions() table
# Refresh button to re-fetch live data
# Deploy on Streamlit Cloud (requires public repo) — see Q5
```

Tasks:
- [x] **`monitoring/alerts.py`** — implement `DiscordAlerter`
- [x] **`monitoring/dashboard.py`** — implement Streamlit app
- [ ] Deploy dashboard to Streamlit Cloud

### Verify Phase 11
```bash
# Alerts
python -c "
import config
from monitoring.alerts import DiscordAlerter
a = DiscordAlerter(config.DISCORD_WEBHOOK_URL)
a.send_error('Phase 11 test — ignore')
"
# → Check Discord for test message

# Dashboard
streamlit run monitoring/dashboard.py
# → Open localhost:8501, confirm all 5 sections render without errors
```

---

## Phase 12 — Job Orchestration

### `jobs/daily_job.py` (NEW)
```python
# Execution order (runs every market day at 09:35 EST):
# 1.  broker.cancel_all_orders()
# 2.  if not broker.is_market_open(): sys.exit(0)
# 3.  LIVE_TRADING gate: if config.LIVE_TRADING is True, print warning
#     and require CONFIRM=true env var before proceeding — prevents accidents
# 4.  FOR EACH ticker IN config.WATCHLIST:
#   a. bar = feed.get_latest_bar(ticker)
#      db.upsert_bars(DataFrame([bar]), ticker)
#   b. score = sentiment.get_daily_score(ticker, today)
#   c. df = db.get_bars(ticker, lookback_start, today)
#      sentiment_df = db.get_sentiment(ticker, lookback_start, today)
#   d. model, _ = registry.load_best(metric='rmse', name_prefix=f'ensemble_{ticker}')
#   e. X, _, _ = get_features_at(df, today)
#      X = merge_sentiment(X_as_df, sentiment_df)
#   f. predicted = model.predict(X[-1:])
#      confidence = model.get_confidence(X[-1:]) if hasattr(model,'get_confidence') else 0.7
#   g. signal = signal_gen.generate(bar['close'], predicted[0], confidence)
#   h. order = order_manager.execute_signal(signal, ticker, df)
# 5. portfolio.snapshot(today)
# 6. alerts.send_daily_summary(signals_dict, portfolio.get_portfolio_value())
#    signals_dict: {ticker: signal} for all 3 tickers
```

### `jobs/train_job.py` (NEW)
```python
# Execution order (runs every Sunday at 02:00 UTC):
# FOR EACH ticker IN config.WATCHLIST:
# 1. df = db.get_bars(ticker, lookback_start, today)
# 2. sentiment_df = db.get_sentiment(ticker, lookback_start, today)
# 3. _, current_best = registry.load_best('rmse', name_prefix=f'ensemble_{ticker}')
# 4. metrics = WalkForwardTrainer().train(df, sentiment_df, ticker=ticker)
#    (trainer saves to registry as f'ensemble_{ticker}' if RMSE improved)
# 5. alerts.send_retrain_summary({ticker: metrics})
# NOTE: Models are stored per-ticker. registry.load_best() accepts
# name_prefix kwarg to filter by ticker. Add this to ModelRegistry.load_best().
```

Tasks:
- [ ] **`jobs/daily_job.py`** — implement with `--dry-run` flag support
- [ ] **`jobs/train_job.py`** — implement

### Verify Phase 12
```bash
python jobs/daily_job.py --dry-run
# → Prints each step, sends Discord summary, does NOT submit real order
python jobs/train_job.py
# → Prints fold metrics, sends Discord summary
```

---

## Phase 13 — Scheduling & Deployment

### `.github/workflows/daily_trade.yml` (NEW)
```yaml
name: Daily Trade Job
on:
  schedule:
    - cron: '35 14 * * 1-5'   # 09:35 EST = 14:35 UTC, Mon–Fri
  workflow_dispatch:            # Manual trigger for testing
jobs:
  trade:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python jobs/daily_job.py
    env:
      ALPACA_API_KEY:      ${{ secrets.ALPACA_API_KEY }}
      ALPACA_SECRET_KEY:   ${{ secrets.ALPACA_SECRET_KEY }}
      ALPACA_BASE_URL:     ${{ secrets.ALPACA_BASE_URL }}
      NEWS_API_KEY:        ${{ secrets.NEWS_API_KEY }}
      DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
      DB_URL:              ${{ secrets.DB_URL }}   # Turso/Neon connection string
```

### `.github/workflows/weekly_retrain.yml` (NEW)
```yaml
name: Weekly Retrain
on:
  schedule:
    - cron: '0 2 * * 0'   # 02:00 UTC every Sunday
  workflow_dispatch:
jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python jobs/train_job.py
    env:
      ALPACA_API_KEY:      ${{ secrets.ALPACA_API_KEY }}
      ALPACA_SECRET_KEY:   ${{ secrets.ALPACA_SECRET_KEY }}
      NEWS_API_KEY:        ${{ secrets.NEWS_API_KEY }}
      DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
      DB_URL:              ${{ secrets.DB_URL }}
```

### Updated `requirements.txt` — add these lines
```
alpaca-trade-api==3.3.2
sqlalchemy==2.0.0
psycopg2-binary==2.9.9
python-dotenv==1.0.0
nltk==3.8.1
newsapi-python==0.2.7
streamlit==1.32.0
requests==2.31.0
```
`psycopg2-binary` is required for Neon.tech (PostgreSQL). The `-binary` variant
bundles its own libpq — no system Postgres installation needed on the runner.

Tasks:
- [ ] **`.github/workflows/daily_trade.yml`** — create workflow file
- [ ] **`.github/workflows/weekly_retrain.yml`** — create workflow file
- [ ] Add all secrets to GitHub repo Settings → Secrets → Actions:
      `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`,
      `NEWS_API_KEY`, `DISCORD_WEBHOOK_URL`, `DB_URL`
      Do NOT add `LIVE_TRADING=true` — leave it absent (defaults to false)
- [ ] **`requirements.txt`** — add new dependencies
- [ ] Deploy `monitoring/dashboard.py` to Streamlit Cloud
      (streamlit.io/cloud → "New app" → point to this repo)
- [ ] Trigger `workflow_dispatch` manual run to verify end-to-end before
      relying on the cron schedule
- [ ] **LIVE-MONEY BLOCKER**: Before setting `LIVE_TRADING=true` anywhere:
      (a) enable Streamlit Community Cloud Google OAuth on the dashboard app
      (b) confirm 3+ months of clean paper trading with no crashes
      (c) confirm Alpaca account is configured as CASH (not margin) to avoid PDT rule

### Verify Phase 13
```
1. Push .github/workflows/ to main branch
2. GitHub → Actions → "Daily Trade Job" → "Run workflow" (manual trigger)
3. Watch logs for each of the 12 steps in daily_job.py
4. Confirm Discord alert received
5. Confirm DB trade log updated (check via dashboard or sqlite3 CLI)
```

---

## Dependency Graph

Must be completed in this order (each phase requires all listed predecessors):

```
Phase 1  (Accounts)      → no dependencies
Phase 2  (Restructure)   → Phase 1
Phase 3  (Config/DB)     → Phase 2
Phase 4  (Data Layer)    → Phase 3
Phase 5  (Features)      → Phase 3, Phase 4
Phase 6  (Models)        → Phase 5
Phase 7  (Training)      → Phase 5, Phase 6
Phase 8  (Signals/Risk)  → Phase 4, Phase 7
Phase 9  (Execution)     → Phase 8
Phase 10 (Backtest)      → Phase 5, Phase 6, Phase 8
Phase 11 (Monitoring)    → Phase 3, Phase 9
Phase 12 (Jobs)          → Phases 4–9, Phase 11
Phase 13 (Deploy)        → Phase 12
```

---

## Decisions Made (2026-04-04)

All five open questions are answered. Do not re-open these without discussion.

| # | Decision | Notes |
|---|---|---|
| Q1 | **Neon.tech (PostgreSQL)** | Most robust free option. Add `psycopg2-binary` to requirements. Neon auto-suspends after 5 min idle — cold start adds ~500ms, fine for daily jobs. |
| Q2 | **Daily EOD signals** | Runs once at 09:35 EST. Fits all free-tier API rate limits. Simpler to reason about and debug. |
| Q3 | **Small watchlist: AAPL, MSFT, GOOGL** | One model trained per ticker (stored in registry as `ensemble_AAPL` etc.). All three are correlated tech stocks — not true diversification, but acceptable for this project. `MAX_POSITION_PCT` set to 15% (not 20%) to cap total tech exposure at 45%. |
| Q4 | **Eventually real money** | Repo public repo is fine. Use **cash account** on Alpaca (not margin) to avoid PDT rule. `LIVE_TRADING=false` env var required — must be explicitly set `true` with a printed confirmation before any live order fires. Minimum 3 months of clean paper trading before going live. |
| Q5 | **Public repo + Streamlit Cloud** | Public is fine for paper trading (good portfolio piece). Before switching to real money: enable Streamlit Community Cloud Google OAuth to password-protect dashboard. This is a **hard blocker** for Phase 13 live-money transition. |

---

## Handoff Notes

<!-- Append entries here as implementation proceeds. Newest first. -->

[2026-04-04] Phases 2 & 3 complete. Created 9 package dirs, config.py,
             data/database.py (SQLAlchemy Core + pg_insert upserts),
             features/indicators.py (extracted from StockDataLoader).
             StockDataLoader.add_technical_indicators() now delegates to
             features.indicators.add_indicators() — no behavior change.
             Next: Phase 5 remainder (walk_forward.py, sentiment_features.py)
             then Phase 4 (alpaca_feed.py, news_sentiment.py).

[2026-04-04] All 5 open questions answered. Key decisions locked in:
             - Database: Neon.tech PostgreSQL (DB_URL env var, psycopg2-binary)
             - Frequency: daily EOD at 09:35 EST
             - Watchlist: AAPL, MSFT, GOOGL — one model per ticker stored as
               ensemble_<TICKER> in registry; MAX_POSITION_PCT lowered to 15%
             - Live money: cash account on Alpaca (avoid PDT); LIVE_TRADING env
               var gate added to config.py; 3-month paper validation required
             - Dashboard: public repo + Streamlit Cloud; Streamlit OAuth required
               before going live (hard blocker in Phase 13)
             Phase 1 (account creation) is now unblocked.

[2026-04-03] ROADMAP upgraded to Cursor-ready living spec. Added: codebase
             inventory with full method signatures, inline specs for all 19 new
             files, dependency graph, per-phase verify commands, this log.
             Open questions Q1–Q5 not yet answered. Implementation blocked
             on Q1 (database choice determines DB_URL format throughout).
