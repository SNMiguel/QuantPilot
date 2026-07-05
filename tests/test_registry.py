"""Model registry: metadata filters and latest-version loading."""
import numpy as np
from sklearn.linear_model import LinearRegression

from models.registry import ModelRegistry


def make_registry(tmp_path):
    return ModelRegistry(registry_path=str(tmp_path / "registry.json"))


def fitted_model(seed=0):
    rng = np.random.default_rng(seed)
    return LinearRegression().fit(rng.normal(size=(50, 4)),
                                  rng.normal(size=50))


def test_save_and_load_latest(tmp_path):
    reg = make_registry(tmp_path)
    reg.save(fitted_model(1), 'ensemble_AAPL', {'rmse': 0.010},
             meta={'target': 'next_return', 'n_features': 4})
    reg.save(fitted_model(2), 'ensemble_AAPL', {'rmse': 0.012},
             meta={'target': 'next_return', 'n_features': 4})

    model, entry = reg.load_latest('ensemble_AAPL',
                                   require_meta={'target': 'next_return'})
    assert model is not None
    # Latest saved wins, even though its stored rmse is higher —
    # promotion quality is decided at train time on a shared window.
    assert entry['metrics']['rmse'] == 0.012


def test_meta_filter_excludes_incompatible_targets(tmp_path):
    """A legacy price-level model must never be served as a return model."""
    reg = make_registry(tmp_path)
    reg.save(fitted_model(), 'ensemble_AAPL', {'rmse': 1.5})  # no meta

    model, entry = reg.load_latest('ensemble_AAPL',
                                   require_meta={'target': 'next_return'})
    assert model is None and entry is None


def test_prefix_isolation(tmp_path):
    reg = make_registry(tmp_path)
    reg.save(fitted_model(), 'ensemble_MSFT', {'rmse': 0.01},
             meta={'target': 'next_return'})

    model, entry = reg.load_latest('ensemble_AAPL',
                                   require_meta={'target': 'next_return'})
    assert model is None
