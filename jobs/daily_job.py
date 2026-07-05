"""
Daily trading job — runs once per day after market close.

Execution order:
  1.  Cancel any leftover open orders
  2.  Check today was a trading day (calendar, not clock — the job runs
      after the close, so the clock always says "closed")
  3.  Circuit breaker: halt if portfolio drawdown exceeds the cap
  4.  Fetch latest price bars for each ticker (cached in DB)
  5.  Fetch today's news sentiment for each ticker (cached in DB)
  6.  Build leakage-free features up to today
  7.  Load the latest promoted model per ticker (next-return target only)
  8.  Predict next-day return, compute confidence
  9.  Generate signal (BUY / SELL / HOLD), log prediction to DB
  10. Execute signal via OrderManager (orders queue for next open)
  11. Snapshot portfolio value in DB
  12. Send daily summary to Discord

Usage:
    python -m jobs.daily_job
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import traceback

import config
from data.database import Database
from data.alpaca_feed import AlpacaFeed
from data.llm_sentiment import LLMSentiment
from features.walk_forward import get_latest_features
from features.sentiment_features import merge_sentiment
from features.regime import detect_regime
from models.registry import ModelRegistry
from signals.generator import SignalGenerator
from risk.position_sizer import PositionSizer
from risk.portfolio import Portfolio
from execution.alpaca_broker import AlpacaBroker
from execution.order_manager import OrderManager
from monitoring.alerts import DiscordAlerter
from monitoring.narrator import TradeNarrator
from training.walk_forward_trainer import TARGET_KIND


def run() -> None:
    today = date.today().isoformat()
    print(f"\n{'='*55}")
    print(f"  Daily Job — {today}")
    print(f"  Live trading: {config.LIVE_TRADING}")
    print(f"{'='*55}")

    alerter = DiscordAlerter(config.DISCORD_WEBHOOK_URL)

    try:
        # ------------------------------------------------------------------
        # 1. Connections
        # ------------------------------------------------------------------
        db      = Database(config.DB_URL)
        db.create_tables()
        feed    = AlpacaFeed(config.ALPACA_API_KEY,
                             config.ALPACA_SECRET_KEY,
                             config.ALPACA_BASE_URL)
        broker  = AlpacaBroker(config.ALPACA_API_KEY,
                               config.ALPACA_SECRET_KEY,
                               config.ALPACA_BASE_URL)
        sizer   = PositionSizer(max_position_pct=config.MAX_POSITION_PCT)
        portfolio = Portfolio(feed, db)
        order_mgr = OrderManager(broker, portfolio, sizer, db, alerter)
        registry  = ModelRegistry()
        sentiment_client = LLMSentiment(
            config.NEWS_API_KEY,
            model=config.LLM_MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
        )
        narrator = TradeNarrator(
            model=config.LLM_MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
        )

        # ------------------------------------------------------------------
        # 2. Cancel stale orders
        # ------------------------------------------------------------------
        broker.cancel_all_orders()
        print("Cancelled any open orders")

        # ------------------------------------------------------------------
        # 3. Trading-day check (calendar, not clock)
        # ------------------------------------------------------------------
        if not broker.market_traded_today():
            print("No trading session today (weekend/holiday) — exiting.")
            return

        # ------------------------------------------------------------------
        # 4. Circuit breaker
        # ------------------------------------------------------------------
        drawdown = portfolio.get_max_drawdown()
        if drawdown > config.MAX_DRAWDOWN_HALT:
            msg = (f"Circuit breaker: drawdown {drawdown:.1%} exceeds "
                   f"{config.MAX_DRAWDOWN_HALT:.0%} cap. No trades today — "
                   f"manual review required.")
            print(msg)
            alerter.send_error(msg)
            portfolio.snapshot(today)
            return

        # ------------------------------------------------------------------
        # 5–10. Per-ticker loop
        # ------------------------------------------------------------------
        all_signals: dict = {}
        all_context: dict = {}   # per-ticker inputs for the trade narrator

        for ticker in config.WATCHLIST:
            print(f"\n--- {ticker} ---")

            # 5a. Fetch price history (cached in DB)
            start_date = "2022-01-01"
            df = feed.get_historical_bars(ticker, start_date, today, db=db)
            if df.empty:
                print(f"  WARN: no price data — skipping {ticker}")
                continue

            # 5b. Fetch sentiment (LLM verdict; VADER fallback)
            sentiment_analysis = sentiment_client.get_daily_analysis(ticker, today)
            sentiment_score    = sentiment_analysis['score']
            db.upsert_sentiment(today, ticker, sentiment_score)
            sentiment_df       = db.get_sentiment(ticker, start_date, today)
            events = ", ".join(sentiment_analysis['key_events'][:3]) or "no material events"
            print(f"  Sentiment today: {sentiment_score:+.4f} "
                  f"[{sentiment_analysis['source']}] ({events})")

            # 6. Latest feature row (leakage-free, includes today)
            feature_df = get_latest_features(df, today)
            if feature_df.empty:
                print(f"  WARN: not enough feature rows — skipping {ticker}")
                continue
            feature_df = merge_sentiment(feature_df, sentiment_df)

            # 7. Load the latest promoted model for this ticker.
            # The target filter refuses models trained to predict price
            # levels — treating those outputs as returns would be
            # catastrophic.
            model, meta = registry.load_latest(
                f'ensemble_{ticker}',
                require_meta={'target': TARGET_KIND},
            )
            if model is None:
                print(f"  WARN: no {TARGET_KIND} model for {ticker} — "
                      f"run train_job first")
                continue
            if meta.get('meta', {}).get('n_features') != feature_df.shape[1]:
                print(f"  WARN: feature count mismatch for {ticker} — "
                      f"retrain required, skipping")
                continue
            print(f"  Model {meta['version_id']}  "
                  f"RMSE={meta['metrics']['rmse']:.5f}  "
                  f"dir_acc={meta['metrics'].get('dir_acc', 0):.3f}")

            # 8. Predict next-day return on today's row
            X_today          = feature_df.values[-1].reshape(1, -1)
            predicted_return = float(model.predict(X_today)[0])
            current          = float(df['Close'].iloc[-1])

            confidence = 0.5
            if hasattr(model, 'get_confidence'):
                confidence = float(model.get_confidence(X_today))

            print(f"  Current: ${current:.2f}  "
                  f"Predicted return: {predicted_return:+.3%}  "
                  f"Conf: {confidence:.2f}")

            # 9. Generate signal and log the prediction
            threshold  = config.SIGNAL_THRESHOLD_OVERRIDES.get(
                ticker, config.SIGNAL_THRESHOLD)
            signal_gen = SignalGenerator(
                threshold=threshold,
                confidence_threshold=config.CONFIDENCE_THRESHOLD,
            )
            signal = signal_gen.generate_from_return(
                current, predicted_return, confidence)

            # Conformal gate: veto BUY/SELL unless the prediction interval
            # excludes zero (the move is distinguishable from noise).
            if (config.USE_CONFORMAL_GATE and signal['signal'] != 'HOLD'
                    and hasattr(model, 'interval_excludes_zero')):
                try:
                    directional = bool(
                        model.interval_excludes_zero(
                            X_today, config.CONFORMAL_COVERAGE)[0])
                    if not directional:
                        print(f"  Conformal gate: {config.CONFORMAL_COVERAGE:.0%} "
                              f"interval includes zero — downgrading "
                              f"{signal['signal']} to HOLD")
                        signal['signal'] = 'HOLD'
                except Exception as exc:
                    print(f"  Note: conformal gate skipped ({exc})")

            all_signals[ticker] = signal
            print(f"  Signal: {signal['signal']}  d{signal['delta_pct']:+.2f}%")

            db.upsert_prediction(
                date=today,
                ticker=ticker,
                model_version=meta['version_id'],
                predicted_price=signal['predicted'],
                signal=signal['signal'],
                confidence=confidence,
            )

            # 10. Volatility regime → position-size scaling
            regime = detect_regime(df)
            if regime['regime'] not in ('calm', 'normal'):
                print(f"  Regime: {regime['regime']} "
                      f"(vol pct {regime['percentile']:.0%}) → "
                      f"size x{regime['size_multiplier']:.2f}")

            # 11. Execute
            order = order_mgr.execute_signal(
                signal, ticker, df,
                size_multiplier=regime['size_multiplier'])
            if order:
                print(f"  Order submitted: {order.get('id', order)}")
            else:
                print(f"  No order placed (HOLD, flat, or risk limit)")

            # Record inputs for the end-of-day narration
            all_context[ticker] = {
                'signal':           signal['signal'],
                'predicted_return': predicted_return,
                'confidence':       confidence,
                'sentiment':        sentiment_analysis,
                'regime':           regime['regime'],
                'order':            order is not None,
                'blocked':          signal['signal'] != 'HOLD' and order is None,
            }

        # ------------------------------------------------------------------
        # 11. Snapshot portfolio
        # ------------------------------------------------------------------
        portfolio_value = portfolio.get_portfolio_value()
        portfolio.snapshot(today)
        print(f"\nPortfolio value: ${portfolio_value:,.2f}")

        # ------------------------------------------------------------------
        # 12. Narrate + Discord summary
        # ------------------------------------------------------------------
        rationale = narrator.narrate(portfolio_value, all_context)
        print(f"\nRationale: {rationale}")
        alerter.send_daily_summary(all_signals, portfolio_value, rationale=rationale)
        print("Discord summary sent")
        print(f"\n{'='*55}")
        print("  Daily job complete")
        print(f"{'='*55}\n")

    except Exception as exc:
        msg = f"daily_job failed: {exc}\n{traceback.format_exc()}"
        print(f"\nERROR: {msg}")
        alerter.send_error(msg)
        raise


if __name__ == "__main__":
    run()
