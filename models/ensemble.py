"""
Ensemble model — stacks base model predictions using a Ridge meta-learner.

The meta-learner is trained on out-of-fold predictions produced with
sklearn's TimeSeriesSplit, so every OOF prediction comes from a model
that saw only PAST data. After the OOF pass, the base models are refit
on the full training window so production predictions use all data.

Confidence is directional agreement: the fraction of base models whose
predicted sign matches the ensemble's predicted sign. With three base
models this yields 1.0, 0.67, or 0.33 — a gate at 0.60 means "at least
two of three models agree on direction".
"""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR


def make_base_models() -> dict:
    """
    Fresh, unfitted base estimators tuned for a NEXT-DAY RETURN target
    (values on the order of 0.01, not price levels — hence the small
    SVR epsilon; the old epsilon=0.1 would predict a constant).
    """
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.0005),
    }


class EnsembleModel:
    """Stacking ensemble with a leakage-free meta-learner."""

    def __init__(self, base_models: dict = None):
        """
        Args:
            base_models: Dict mapping model name -> sklearn estimator
                         (fitted or not — fit() clones and refits them).
                         Defaults to make_base_models().
        """
        self.base_models  = base_models if base_models is not None \
                            else make_base_models()
        # alpha must be tiny: the meta-features are return-scale (~0.005),
        # so X'X entries are ~1e-5 per sample. alpha=1.0 (the price-scale
        # default) would dominate the normal equations and shrink the
        # meta-learner to a constant.
        self.meta_learner = Ridge(alpha=1e-4)
        self.is_fitted    = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            n_splits: int = 5) -> None:
        """
        Fit the meta-learner on out-of-fold predictions, then refit all
        base models on the full training window.

        TimeSeriesSplit guarantees each validation chunk is strictly
        AFTER its training data, so no future information reaches the
        meta-learner. Cloned estimators are used for the OOF pass so the
        production base models are never left fitted on a fold subset.
        """
        names = list(self.base_models.keys())
        tscv  = TimeSeriesSplit(n_splits=n_splits)

        oof_blocks, y_blocks = [], []
        for train_idx, val_idx in tscv.split(X_train):
            fold_preds = []
            for name in names:
                m = clone(self.base_models[name])
                m.fit(X_train[train_idx], y_train[train_idx])
                fold_preds.append(np.asarray(m.predict(X_train[val_idx])).flatten())
            oof_blocks.append(np.column_stack(fold_preds))
            y_blocks.append(y_train[val_idx])

        self.meta_learner.fit(np.vstack(oof_blocks), np.concatenate(y_blocks))

        # Production base models: refit on everything
        for name in names:
            self.base_models[name].fit(X_train, y_train)

        self.is_fitted = True
        print(f"Ensemble fitted ({n_splits} time-series folds, "
              f"{len(names)} base models)")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict next-day returns, shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return self.meta_learner.predict(self._stack_predictions(X))

    def get_confidence(self, X: np.ndarray) -> float:
        """
        Directional agreement between base models and the ensemble.

        Returns a scalar in [0, 1]: the mean (over rows) fraction of
        base models whose predicted sign matches the meta-learner's
        predicted sign. 1.0 = unanimous direction.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before get_confidence().")

        stacked   = self._stack_predictions(X)              # (n, n_models)
        ensemble  = self.meta_learner.predict(stacked)      # (n,)
        agreement = (np.sign(stacked) ==
                     np.sign(ensemble)[:, None]).mean(axis=1)
        return float(agreement.mean())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stack_predictions(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_models) array of base model predictions."""
        preds = [np.asarray(m.predict(X)).flatten()
                 for m in self.base_models.values()]
        return np.column_stack(preds)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, f = 400, 15
    X = rng.normal(size=(n, f))
    # Return-scale target (~1% daily moves)
    y = 0.004 * X[:, 0] + 0.002 * X[:, 1] + rng.normal(0, 0.01, n)

    split = int(0.8 * n)
    ensemble = EnsembleModel()
    ensemble.fit(X[:split], y[:split], n_splits=4)

    preds = ensemble.predict(X[split:])
    conf  = ensemble.get_confidence(X[split:split + 1])

    from sklearn.metrics import mean_squared_error
    rmse    = float(np.sqrt(mean_squared_error(y[split:], preds)))
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y[split:])))

    print(f"RMSE (returns)        : {rmse:.5f}")
    print(f"Directional accuracy  : {dir_acc:.3f}")
    print(f"Confidence (1 row)    : {conf:.3f}")
    print("models/ensemble.py: OK")
