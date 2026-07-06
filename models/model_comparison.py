"""
Compare performance across all models (scikit-learn and TensorFlow).
"""
import numpy as np
from .linear_regression import TraditionalModels
from .neural_network import NeuralNetworkModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.evaluation import ModelEvaluator

class ModelComparison:
    """Compare all models and find the best performer."""
    
    def __init__(self):
        self.traditional_models = TraditionalModels()
        self.nn_model = None
        self.evaluator = ModelEvaluator()
        self.results = {}
        
    def train_all_models(self, X_train, y_train, input_dim):
        """
        Train all models (traditional ML + neural network).
        
        Args:
            X_train: Training features
            y_train: Training targets
            input_dim: Number of input features
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        # Train traditional models
        self.traditional_models.train_all(X_train, y_train)
        
        # Train neural network
        print("\n" + "="*50)
        print("Training Deep Learning Model")
        print("="*50)
        self.nn_model = NeuralNetworkModel(input_dim=input_dim)
        self.nn_model.build_model(architecture='standard')
        
        # For neural network, we need to scale the data
        X_train_scaled = self.nn_model.scaler.fit_transform(X_train)
        self.nn_model.train(X_train_scaled, y_train, epochs=100, verbose=0)
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*60 + "\n")
    
    def evaluate_all_models(self, X_test, y_test, dates_test):
        """
        Evaluate all models on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            dates_test: Test dates for plotting
        """
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60 + "\n")
        
        # Evaluate traditional models
        for model_name in self.traditional_models.models.keys():
            y_pred = self.traditional_models.predict(model_name, X_test)
            metrics = self.evaluator.calculate_metrics(y_test, y_pred, model_name)
            self.evaluator.print_metrics(metrics, model_name)
            self.results[model_name] = {
                'predictions': y_pred,
                'metrics': metrics
            }
        
        # Evaluate neural network
        X_test_scaled = self.nn_model.scaler.transform(X_test)
        y_pred_nn = self.nn_model.predict(X_test_scaled)
        metrics_nn = self.evaluator.calculate_metrics(y_test, y_pred_nn, "Neural Network (TensorFlow)")
        self.evaluator.print_metrics(metrics_nn, "Neural Network (TensorFlow)")
        self.results["Neural Network (TensorFlow)"] = {
            'predictions': y_pred_nn,
            'metrics': metrics_nn
        }
        
        print("="*60)
        print("EVALUATION COMPLETE")
        print("="*60 + "\n")
    
    def print_comparison(self):
        """Print detailed comparison of all models."""
        self.evaluator.print_comparison_table()
    
    def plot_all_predictions(self, dates_test, y_test, save_dir='results'):
        """
        Plot predictions for all models.
        
        Args:
            dates_test: Test dates
            y_test: Actual test values
            save_dir: Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, result in self.results.items():
            self.evaluator.plot_predictions(
                dates_test, 
                y_test, 
                result['predictions'],
                model_name=model_name,
                save_path=f"{save_dir}/{model_name.replace(' ', '_').lower()}_predictions.png"
            )
    
    def get_best_model(self):
        """Identify the best performing model based on RMSE."""
        best_model = None
        best_rmse = float('inf')
        
        for model_name, result in self.results.items():
            rmse = result['metrics']['RMSE']
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name
        
        print("\n" + "="*60)
        print("BEST MODEL")
        print("="*60)
        print(f"{best_model}")
        print(f"   RMSE: ${best_rmse:.2f}")
        print(f"   R2 Score: {self.results[best_model]['metrics']['R2']:.4f}")
        print("="*60 + "\n")
        
        return best_model, self.results[best_model]


if __name__ == "__main__":
    # Test model comparison
    print("Testing Model Comparison...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 18
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(n_samples) * 2
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Train all models
    comparison.train_all_models(X_train, y_train, input_dim=n_features)
    
    # Evaluate
    import pandas as pd
    dates_test = pd.date_range('2023-01-01', periods=len(y_test))
    comparison.evaluate_all_models(X_test, y_test, dates_test)
    
    # Print comparison
    comparison.print_comparison()
    
    # Get best model
    best_model, best_results = comparison.get_best_model()
    
    print("Model comparison test complete!")