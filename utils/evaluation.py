"""
Evaluation metrics and visualization utilities for stock prediction models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ModelEvaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self):
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """
        Calculate regression metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model for storage
            
        Returns:
            dict: Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, metrics, model_name="Model"):
        """Print metrics in a formatted way."""
        print(f"\n{'='*50}")
        print(f"{model_name} Performance Metrics")
        print(f"{'='*50}")
        print(f"Mean Absolute Error (MAE):       ${metrics['MAE']:.2f}")
        print(f"Mean Squared Error (MSE):        ${metrics['MSE']:.2f}")
        print(f"Root Mean Squared Error (RMSE):  ${metrics['RMSE']:.2f}")
        print(f"R2 Score:                        {metrics['R2']:.4f}")
        print(f"Mean Absolute Percentage Error:  {metrics['MAPE']:.2f}%")
        print(f"{'='*50}\n")
    
    def plot_predictions(self, dates, y_true, y_pred, model_name="Model", save_path=None):
        """
        Plot actual vs predicted prices.
        
        Args:
            dates: Date index
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name for plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(14, 6))
        
        plt.plot(dates, y_true, label='Actual Price', color='#2E86AB', linewidth=2, alpha=0.8)
        plt.plot(dates, y_pred, label='Predicted Price', color='#A23B72', linewidth=2, linestyle='--', alpha=0.8)
        
        plt.title(f'{model_name}: Actual vs Predicted Stock Prices', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, model_name="Model", save_path=None):
        """
        Plot residuals (prediction errors).
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name for plot title
            save_path: Optional path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals over predictions
        axes[0].scatter(y_pred, residuals, alpha=0.5, color='#2E86AB')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title(f'{model_name}: Residual Plot', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'{model_name}: Residuals Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, save_path=None):
        """
        Compare metrics across all evaluated models.
        
        Args:
            save_path: Optional path to save the comparison plot
        """
        if not self.results:
            print("No models to compare. Evaluate models first.")
            return
        
        models = list(self.results.keys())
        metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            
            bars = axes[idx].bar(models, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(models)], alpha=0.8)
            axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def print_comparison_table(self):
        """Print a formatted comparison table of all models."""
        if not self.results:
            print("No models to compare. Evaluate models first.")
            return
        
        print(f"\n{'='*80}")
        print(f"{'Model Comparison Table':^80}")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R2':<12} {'MAPE':<12}")
        print(f"{'-'*80}")
        
        for model_name, metrics in self.results.items():
            print(f"{model_name:<20} ${metrics['MAE']:<11.2f} ${metrics['RMSE']:<11.2f} "
                  f"{metrics['R2']:<12.4f} {metrics['MAPE']:<11.2f}%")
        
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test the evaluator
    np.random.seed(42)
    
    # Generate sample data
    y_true = np.random.uniform(100, 200, 100)
    y_pred1 = y_true + np.random.normal(0, 5, 100)  # Good predictions
    y_pred2 = y_true + np.random.normal(0, 15, 100)  # Worse predictions
    dates = pd.date_range('2023-01-01', periods=100)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    metrics1 = evaluator.calculate_metrics(y_true, y_pred1, "Model A")
    evaluator.print_metrics(metrics1, "Model A")
    
    metrics2 = evaluator.calculate_metrics(y_true, y_pred2, "Model B")
    evaluator.print_metrics(metrics2, "Model B")
    
    # Compare
    evaluator.print_comparison_table()