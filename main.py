"""
LEGACY DEMO — not the production trading path.

This is the original single-file model-comparison demo (predicts the
SAME-day AAPL close from full-dataset indicators). It is retained only to
show the project's starting point; it contains the look-ahead bias the
production pipeline was built to fix. Do NOT use it for trading or as a
template — the live system runs through jobs/ (see README.md / CLAUDE.md).

Kept runnable for historical reference: python main.py
"""
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import StockDataLoader
from models.model_comparison import ModelComparison
import numpy as np
from utils.sample_data import generate_sample_aapl_data

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print(" "*15 + "AAPL STOCK PRICE PREDICTION")
    print(" "*10 + "Multi-Framework ML Model Comparison")
    print("="*70 + "\n")
    
    # Step 1: Load and prepare data
    print("STEP 1: Loading AAPL Stock Data")
    print("-"*70)
    
    loader = StockDataLoader(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2024-01-01"
    )
    
    # Download data
    loader.download_data()
    
    # Add technical indicators
    loader.add_technical_indicators()
    
    # Prepare features
    X, y, dates = loader.prepare_features()
    
    print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Price range: ${y.min():.2f} - ${y.max():.2f}")
    print(f"Features: {', '.join(loader.feature_names[:5])}... (and {len(loader.feature_names)-5} more)")
    
    # Step 2: Split data (80% train, 20% test)
    print("\n" + "="*70)
    print("STEP 2: Splitting Data (80% Train, 20% Test)")
    print("-"*70)
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training period: {dates_train[0].date()} to {dates_train[-1].date()}")
    print(f"Test period: {dates_test[0].date()} to {dates_test[-1].date()}")
    
    # Step 3: Train all models
    print("\n" + "="*70)
    print("STEP 3: Training All Models")
    print("-"*70)
    
    comparison = ModelComparison()
    comparison.train_all_models(X_train, y_train, input_dim=X.shape[1])
    
    # Step 4: Evaluate all models
    print("\n" + "="*70)
    print("STEP 4: Evaluating Models on Test Data")
    print("-"*70)
    
    comparison.evaluate_all_models(X_test, y_test, dates_test)
    
    # Step 5: Compare and identify best model
    print("\n" + "="*70)
    print("STEP 5: Model Comparison & Results")
    print("-"*70)
    
    comparison.print_comparison()
    best_model_name, best_results = comparison.get_best_model()
    
    # Step 6: Visualizations (optional - will show plots)
    print("\n" + "="*70)
    print("STEP 6: Generating Visualizations")
    print("-"*70)
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Plot predictions for best model
    print(f"Generating prediction plot for {best_model_name}...")
    comparison.evaluator.plot_predictions(
        dates_test,
        y_test,
        best_results['predictions'],
        model_name=best_model_name,
        save_path=f'results/best_model_predictions.png'
    )
    
    # Plot residuals for best model
    print(f"Generating residual plot for {best_model_name}...")
    comparison.evaluator.plot_residuals(
        y_test,
        best_results['predictions'],
        model_name=best_model_name,
        save_path=f'results/best_model_residuals.png'
    )
    
    # Plot comparison
    print("Generating model comparison plot...")
    comparison.evaluator.compare_models(save_path='results/model_comparison.png')
    
    print("\nAll visualizations saved to 'results/' directory")
    
    # Final summary
    print("\n" + "="*70)
    print(" "*20 + "PROJECT SUMMARY")
    print("="*70)
    print(f"Dataset:           AAPL Stock (2020-2024)")
    print(f"Total Samples:     {len(X)}")
    print(f"Features:          {X.shape[1]} technical indicators")
    print(f"Models Trained:    4 (Linear Regression, Random Forest, SVR, Neural Network)")
    print(f"Best Model:        {best_model_name}")
    print(f"Best RMSE:         ${best_results['metrics']['RMSE']:.2f}")
    print(f"Best R² Score:     {best_results['metrics']['R²']:.4f}")
    print("\nFrameworks Used:")
    print("  • scikit-learn   - Traditional ML models")
    print("  • TensorFlow     - Deep learning neural network")
    print("  • pandas         - Data manipulation")
    print("  • matplotlib     - Visualization")
    print("="*70 + "\n")
    
    print("Project Complete! Check the 'results/' folder for visualizations.")


if __name__ == "__main__":
    main()