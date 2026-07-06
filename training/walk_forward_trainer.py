"""
Walk-forward training pipeline.

Expanding-window validation of the EXACT model that gets deployed (the
stacking ensemble), followed by a production retrain and a fair
promotion decision:

  - Each fold trains a fresh ensemble on data up to a cutoff and
    evaluates it on the next unseen chunk. Fold metrics therefore
    describe the deployed model, not a different one.
  - The target is the next-day return (see features/walk_forward.py),
    so RMSE is in return units and directional accuracy is reported.
  - Promotion: the incumbent registry model and the new challenger are
    both evaluated on the SAME held-out window. The challenger is saved
    only if it wins there - comparing stored metrics from different
    time windows (the old behaviour) rewards whichever model was
    evaluated during a calmer market.
"""
import numpy as np
import pandas as pd

from features.walk_forward import get_features_at
from features.sentiment_features import merge_sentiment
from models.ensemble import EnsembleModel, make_base_models
from models.registry import ModelRegistry
from training.metrics import directional_accuracy

TARGET_KIND = 'next_return'   # stamped into registry meta on every save

# Promotion guardrails. Daily-return RMSE differences between retrains are
# mostly noise, so a challenger must beat the incumbent's RMSE by a real
# relative margin AND not regress on directional accuracy. Without this a
# 0.0001 RMSE flicker could demote a better model on a single window.
_PROMOTION_RMSE_MARGIN = 0.02   # challenger RMSE must be >=2% lower
_PROMOTION_DIRACC_SLACK = 0.01  # allow at most this much dir_acc regression


class WalkForwardTrainer:
    """Expanding-window walk-forward trainer for the stacking ensemble."""

    def __init__(self, n_splits: int = 5,
                 retrain_window_days: int = 500):
        """
        Args:
            n_splits:             Number of walk-forward validation folds.
            retrain_window_days:  How many most-recent trading days to use
                                  for the final production retrain.
        """
        self.n_splits            = n_splits
        self.retrain_window_days = retrain_window_days

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame,
              sentiment_df: pd.DataFrame,
              ticker: str = None) -> dict:
        """
        Run walk-forward validation then retrain on the most recent window.

        Args:
            df:            Raw OHLCV DataFrame with DatetimeIndex.
            sentiment_df:  Sentiment scores DataFrame (date index, 'score'
                           col). Can be empty - sentiment defaults to 0.0.
            ticker:        Ticker symbol e.g. 'AAPL'. Registry key prefix.

        Returns:
            Dict of averaged fold metrics:
            {'rmse': float, 'mae': float, 'r2': float, 'dir_acc': float}
            RMSE/MAE are in RETURN units (0.01 = 1%).
        """
        if df is None or df.empty:
            raise ValueError(
                f"Empty DataFrame for ticker '{ticker}'. "
                "Check data fetch - Alpaca may have fallen back to yfinance "
                "with no result."
            )

        print(f"\n{'='*60}")
        print(f"Walk-Forward Training  -  {ticker or 'unknown'}")
        print(f"Data: {len(df)} rows  |  Folds: {self.n_splits}  |  "
              f"Final window: {self.retrain_window_days} days")
        print('='*60)

        fold_metrics = self._run_folds(df, sentiment_df)

        if not fold_metrics:
            print("WARN: no valid folds completed. Check data length.")
            return {}

        avg = self._average_metrics(fold_metrics)

        print(f"\n{'='*60}")
        print("Walk-Forward Averaged Metrics (ensemble, next-day returns)")
        print('='*60)
        print(f"  RMSE     : {avg['rmse']:.5f}")
        print(f"  MAE      : {avg['mae']:.5f}")
        print(f"  R2       : {avg['r2']:.4f}")
        print(f"  Dir. acc : {avg['dir_acc']:.3f}  (coin flip = 0.500)")

        self._final_retrain(df, sentiment_df, ticker, avg)

        return avg

    # ------------------------------------------------------------------
    # Folds
    # ------------------------------------------------------------------

    def _run_folds(self, df: pd.DataFrame,
                   sentiment_df: pd.DataFrame) -> list:
        """Run n_splits expanding-window folds and return per-fold metrics."""
        dates = df.index
        n     = len(dates)
        step  = n // (self.n_splits + 1)
        fold_results = []

        for i in range(1, self.n_splits + 1):
            train_end_idx = i * step
            test_end_idx  = min((i + 1) * step, n - 1)

            cutoff_train = dates[train_end_idx].strftime('%Y-%m-%d')
            cutoff_test  = dates[test_end_idx].strftime('%Y-%m-%d')

            print(f"\n--- Fold {i}/{self.n_splits} "
                  f"| train -> {cutoff_train}  test -> {cutoff_test} ---")

            X_all, y_all, dates_all = get_features_at(df, cutoff_test)
            X_all = self._merge_sentiment_array(X_all, dates_all, sentiment_df)

            train_mask = dates_all <= cutoff_train
            test_mask  = ~train_mask

            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

            if len(X_train) < 60 or len(X_test) < 10:
                print(f"  WARN: skipping fold {i} - not enough data "
                      f"(train={len(X_train)}, test={len(X_test)})")
                continue

            ensemble = EnsembleModel(make_base_models())
            ensemble.fit(X_train, y_train, n_splits=3)
            preds = ensemble.predict(X_test)

            m = self._score(y_test, preds)
            fold_results.append(m)
            print(f"  Ensemble  RMSE={m['rmse']:.5f}  "
                  f"dir_acc={m['dir_acc']:.3f}  R2={m['r2']:.4f}")

        return fold_results

    # ------------------------------------------------------------------
    # Final retrain + promotion
    # ------------------------------------------------------------------

    def _final_retrain(self, df: pd.DataFrame,
                       sentiment_df: pd.DataFrame,
                       ticker: str,
                       fold_avg: dict) -> None:
        """
        Retrain on the most recent retrain_window_days and promote only
        if the challenger beats the incumbent on the same test window.
        """
        print(f"\n{'='*60}")
        print("Final Production Retrain")
        print('='*60)

        recent_df = df.iloc[-self.retrain_window_days:]
        cutoff    = recent_df.index[-1].strftime('%Y-%m-%d')

        X, y, dates = get_features_at(recent_df, cutoff)
        X = self._merge_sentiment_array(X, dates, sentiment_df)

        split   = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]

        challenger = EnsembleModel(make_base_models())
        challenger.fit(X_train, y_train, n_splits=3)
        chal_metrics = self._score(y_test, challenger.predict(X_test))

        reg_key  = f'ensemble_{ticker}' if ticker else 'ensemble'
        registry = ModelRegistry()

        # Evaluate the incumbent on the SAME window, if one exists and
        # is compatible (same target kind and feature count).
        incumbent, inc_entry = registry.load_latest(
            reg_key, require_meta={'target': TARGET_KIND})
        inc_metrics = None
        if incumbent is not None and \
           inc_entry.get('meta', {}).get('n_features') == X.shape[1]:
            try:
                inc_metrics = self._score(y_test, incumbent.predict(X_test))
            except Exception as exc:
                print(f"  WARN: incumbent could not be evaluated: {exc}")

        print(f"  Challenger : RMSE={chal_metrics['rmse']:.5f}  "
              f"dir_acc={chal_metrics['dir_acc']:.3f}")
        if inc_metrics:
            print(f"  Incumbent  : RMSE={inc_metrics['rmse']:.5f}  "
                  f"dir_acc={inc_metrics['dir_acc']:.3f}")
        else:
            print("  Incumbent  : none (or incompatible) - challenger "
                  "promoted by default")

        promote = inc_metrics is None or self._beats_incumbent(
            chal_metrics, inc_metrics)

        if promote:
            registry.save(
                challenger,
                name=reg_key,
                metrics=chal_metrics,
                framework='sklearn',
                meta={
                    'target':     TARGET_KIND,
                    'n_features': int(X.shape[1]),
                    'test_window': f"{dates[split].date()} -> "
                                   f"{dates[-1].date()}",
                    'fold_avg':   fold_avg,
                    # Conformal interval half-widths (return units) so the
                    # dashboard can render a fan chart without reloading.
                    'conformal_80': challenger.conformal_halfwidth(0.80),
                    'conformal_95': challenger.conformal_halfwidth(0.95),
                },
            )
        else:
            print("  Registry unchanged - challenger did not clear the "
                  "promotion margin.")

    @staticmethod
    def _beats_incumbent(chal: dict, inc: dict) -> bool:
        """
        Promote only on a materially better model, not noise.

        Requires the challenger's RMSE to be at least _PROMOTION_RMSE_MARGIN
        lower (relative) than the incumbent's, and its directional accuracy
        to be no worse than the incumbent's by more than _PROMOTION_DIRACC_SLACK.
        """
        rmse_ok = chal['rmse'] <= inc['rmse'] * (1.0 - _PROMOTION_RMSE_MARGIN)
        diracc_ok = chal['dir_acc'] >= inc['dir_acc'] - _PROMOTION_DIRACC_SLACK
        return rmse_ok and diracc_ok

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        from sklearn.metrics import (mean_absolute_error,
                                     mean_squared_error, r2_score)
        return {
            'rmse':    float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae':     float(mean_absolute_error(y_true, y_pred)),
            'r2':      float(r2_score(y_true, y_pred)),
            'dir_acc': directional_accuracy(y_true, y_pred),
        }

    def _merge_sentiment_array(self, X: np.ndarray,
                               dates: pd.DatetimeIndex,
                               sentiment_df: pd.DataFrame) -> np.ndarray:
        """Attach sentiment scores as an extra trailing column on X."""
        n_features = X.shape[1]
        col_names  = [f'f{i}' for i in range(n_features)]

        feature_df = pd.DataFrame(X, index=dates, columns=col_names)
        merged     = merge_sentiment(feature_df, sentiment_df)
        return merged.values

    @staticmethod
    def _average_metrics(fold_metrics: list) -> dict:
        keys = fold_metrics[0].keys()
        return {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in keys
        }


if __name__ == "__main__":
    import config
    from data.database import Database
    from data.alpaca_feed import AlpacaFeed
    from datetime import date, timedelta

    db   = Database(config.DB_URL)
    feed = AlpacaFeed(config.ALPACA_API_KEY,
                      config.ALPACA_SECRET_KEY,
                      config.ALPACA_BASE_URL)

    ticker = "AAPL"
    end    = date.today().strftime('%Y-%m-%d')
    start  = (date.today() - timedelta(days=config.TRAIN_LOOKBACK_DAYS)
              ).strftime('%Y-%m-%d')

    print(f"Fetching {ticker} data {start} -> {end} ...")
    df           = feed.get_historical_bars(ticker, start, end, db=db)
    sentiment_df = db.get_sentiment(ticker, start, end)

    trainer = WalkForwardTrainer(n_splits=3)
    metrics = trainer.train(df, sentiment_df, ticker=ticker)

    print(f"\nFinal averaged metrics: {metrics}")
    print("training/walk_forward_trainer.py: OK")
