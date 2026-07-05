"""
Versioned model registry.
Saves and loads trained models (sklearn or Keras) with associated metrics.
All versions are tracked in a JSON manifest at MODEL_REGISTRY_PATH.
"""
import os
import json
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np


class ModelRegistry:
    """Save, load, and list versioned trained models."""

    def __init__(self, registry_path: str = None):
        if registry_path is None:
            import config
            registry_path = config.MODEL_REGISTRY_PATH

        self.registry_path = registry_path
        self.registry_dir  = os.path.dirname(registry_path)
        os.makedirs(self.registry_dir, exist_ok=True)

        # Load existing manifest or start fresh
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = []

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, model, name: str, metrics: dict,
             framework: str = 'sklearn', meta: dict = None) -> str:
        """
        Persist a trained model and record it in the manifest.

        Args:
            model:     Fitted sklearn model or Keras model.
            name:      Logical name e.g. 'ensemble_AAPL'.
            metrics:   Dict of evaluation metrics e.g. {'rmse': 3.2, 'r2': 0.95}
            framework: 'sklearn' or 'keras'
            meta:      Optional metadata dict, e.g. {'target': 'next_return',
                       'n_features': 16}. Used by load_latest() filters so a
                       job never loads a model trained for a different target.

        Returns:
            version_id string (uuid4)
        """
        version_id = str(uuid.uuid4())[:8]
        timestamp  = datetime.now(timezone.utc).isoformat()

        if framework == 'keras':
            path = os.path.join(self.registry_dir, f"{version_id}.keras")
            model.save(path)
        else:
            path = os.path.join(self.registry_dir, f"{version_id}.joblib")
            joblib.dump(model, path)

        entry = {
            'version_id': version_id,
            'name':       name,
            'path':       path.replace('\\', '/'),  # always store with forward slashes
            'metrics':    metrics,
            'framework':  framework,
            'timestamp':  timestamp,
            'meta':       meta or {},
        }
        self.manifest.append(entry)
        self._save_manifest()

        print(f"Model saved: {name} [{version_id}]  metrics={metrics}")
        return version_id

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_best(self, metric: str = 'rmse',
                  name_prefix: str = None):
        """
        Load the model with the best (lowest) value for a given metric.

        Args:
            metric:      Key in the metrics dict to optimise (lower = better).
            name_prefix: If set, only consider entries whose name starts with
                         this string e.g. 'ensemble_AAPL'.

        Returns:
            (model, entry_dict) tuple, or (None, None) if registry is empty.
        """
        candidates = self.manifest
        if name_prefix:
            candidates = [e for e in candidates
                          if e['name'].startswith(name_prefix)]

        if not candidates:
            return None, None

        best  = min(candidates,
                    key=lambda e: e['metrics'].get(metric, float('inf')))
        model = self._load_model(best)
        if model is None:
            return None, None
        return model, best

    def load_latest(self, name_prefix: str, require_meta: dict = None):
        """
        Load the most recently saved model matching a name prefix.

        Promotion is decided at train time by comparing incumbent and
        challenger on the SAME test window, so "latest saved" is by
        construction the best available model — unlike load_best(),
        which compares metric values measured on different windows.

        Args:
            name_prefix:  e.g. 'ensemble_AAPL'.
            require_meta: If set, only consider entries whose meta dict
                          contains all these key/value pairs, e.g.
                          {'target': 'next_return'}. Prevents loading a
                          model trained against an incompatible target.

        Returns:
            (model, entry_dict) tuple, or (None, None) if no match.
        """
        candidates = [e for e in self.manifest
                      if e['name'].startswith(name_prefix)]
        if require_meta:
            candidates = [
                e for e in candidates
                if all(e.get('meta', {}).get(k) == v
                       for k, v in require_meta.items())
            ]
        if not candidates:
            return None, None

        latest = max(candidates, key=lambda e: e['timestamp'])
        model  = self._load_model(latest)
        if model is None:
            return None, None
        return model, latest

    def load_version(self, version_id: str):
        """Load a specific model version by its version_id."""
        entry = next((e for e in self.manifest
                      if e['version_id'] == version_id), None)
        if entry is None:
            raise ValueError(f"Version '{version_id}' not found in registry.")
        return self._load_model(entry)

    def _load_model(self, entry: dict):
        path = os.path.normpath(entry['path'])
        try:
            if entry['framework'] == 'keras':
                from tensorflow import keras
                return keras.models.load_model(path)
            else:
                return joblib.load(path)
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_versions(self) -> list:
        """Return all manifest entries sorted by timestamp descending."""
        return sorted(self.manifest,
                      key=lambda e: e['timestamp'], reverse=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_manifest(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)


if __name__ == "__main__":
    import numpy as np
    from sklearn.linear_model import LinearRegression

    reg = ModelRegistry()

    # Save a dummy sklearn model
    m = LinearRegression().fit(np.random.randn(100, 5), np.random.randn(100))
    vid = reg.save(m, 'test_lr', {'rmse': 5.2, 'r2': 0.91}, 'sklearn')

    # List versions
    versions = reg.list_versions()
    print(f"Versions in registry: {len(versions)}")
    for v in versions:
        print(f"  {v['version_id']}  {v['name']}  {v['metrics']}")

    # Load best
    loaded, meta = reg.load_best('rmse')
    print(f"Loaded best: {type(loaded).__name__}  metrics={meta['metrics']}")

    print("models/registry.py: OK")
