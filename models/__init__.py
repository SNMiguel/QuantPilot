from .linear_regression import TraditionalModels
from .registry import ModelRegistry
from .ensemble import EnsembleModel, make_base_models

# Keras models require tensorflow, which is intentionally absent from the
# dashboard and CI environments. Import lazily so `import models` works
# everywhere; NeuralNetworkModel/ModelComparison stay available whenever
# tensorflow is installed.
try:
    from .neural_network import NeuralNetworkModel
    from .model_comparison import ModelComparison
except ImportError:
    NeuralNetworkModel = None
    ModelComparison = None
