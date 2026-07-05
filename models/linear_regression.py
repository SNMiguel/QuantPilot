"""
Linear Regression and traditional ML models for stock prediction.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


class TraditionalModels:
    """Wrapper for traditional ML models using scikit-learn."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_scaled = False
        
    def prepare_data(self, X, y, test_size=0.2, scale=True):
        """
        Split and optionally scale the data.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            scale: Whether to scale features
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Don't shuffle time series
        )
        
        # Scale if requested
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            self.is_scaled = True
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model."""
        print("Training Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['Linear Regression'] = model
        print("Linear Regression trained.")
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """Train Random Forest model."""
        print(f"Training Random Forest ({n_estimators} trees)...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        print("Random Forest trained.")
        return model
    
    def train_svr(self, X_train, y_train):
        """Train Support Vector Regression model."""
        print("Training SVR (this may take a moment)...")
        model = SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.1
        )
        model.fit(X_train, y_train)
        self.models['SVR'] = model
        print("SVR trained.")
        return model
    
    def train_all(self, X_train, y_train):
        """Train all traditional models."""
        print("\n" + "="*50)
        print("Training Traditional ML Models")
        print("="*50)
        
        self.train_linear_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_svr(X_train, y_train)
        
        print("="*50)
        print(f"All {len(self.models)} models trained successfully!")
        print("="*50 + "\n")
    
    def predict(self, model_name, X):
        """Make predictions with a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        return self.models[model_name].predict(X)
    
    def predict_all(self, X):
        """Get predictions from all trained models."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions
    
    def save_models(self, directory='models/saved'):
        """Save all trained models."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filename = f"{directory}/{name.replace(' ', '_').lower()}.joblib"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        # Save scaler if data was scaled
        if self.is_scaled:
            scaler_path = f"{directory}/scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved scaler to {scaler_path}")
    
    def load_models(self, directory='models/saved'):
        """Load previously saved models."""
        import os
        
        model_files = {
            'Linear Regression': 'linear_regression.joblib',
            'Random Forest': 'random_forest.joblib',
            'SVR': 'svr.joblib'
        }
        
        for name, filename in model_files.items():
            filepath = f"{directory}/{filename}"
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"Loaded {name}")
        
        # Load scaler if exists
        scaler_path = f"{directory}/scaler.joblib"
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.is_scaled = True
            print(f"Loaded scaler")


if __name__ == "__main__":
    # Test with sample data
    print("Testing Traditional Models...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(n_samples) * 0.5
    
    # Initialize and train
    trad_models = TraditionalModels()
    X_train, X_test, y_train, y_test = trad_models.prepare_data(X, y, scale=True)
    
    trad_models.train_all(X_train, y_train)
    
    # Get predictions
    predictions = trad_models.predict_all(X_test)
    
    print("\nSample predictions from each model:")
    for name, preds in predictions.items():
        print(f"{name}: {preds[:5]}")
    
    print("\nTraditional models test complete!")