"""
Neural Network models using TensorFlow/Keras for stock prediction.

EXPERIMENTAL - not part of the production ensemble. The live model is the
LR/RF/SVR stack in models/ensemble.py; this Keras LSTM/MLP is kept as a
research track and is imported lazily (models/__init__.py) so the system
runs without TensorFlow. Requires TF, which is optional and unavailable on
Python 3.13+ (see requirements.txt).
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class NeuralNetworkModel:
    """Deep learning model using TensorFlow/Keras."""

    def __init__(self, input_dim, sequence_length=10):
        """
        Initialize the neural network.

        Args:
            input_dim:       Number of input features
            sequence_length: Number of timesteps for LSTM architecture
        """
        self.input_dim       = input_dim
        self.sequence_length = sequence_length
        self.model           = None
        self.scaler          = StandardScaler()
        self.history         = None

    def build_model(self, architecture='standard'):
        """
        Build the neural network architecture.

        Args:
            architecture: 'standard' | 'deep' | 'wide' | 'lstm'
        """
        if architecture == 'lstm':
            # LSTM for sequential pattern learning.
            # Input must be reshaped to (samples, sequence_length, input_dim)
            # before calling train() or predict().
            self.model = keras.Sequential([
                layers.LSTM(64, return_sequences=True,
                            input_shape=(self.sequence_length, self.input_dim)),
                layers.Dropout(0.2),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(1),
            ])

        elif architecture == 'standard':
            # Standard feed-forward network
            self.model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)  # Output layer for regression
            ])
        
        elif architecture == 'deep':
            # Deeper network
            self.model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
        
        elif architecture == 'wide':
            # Wider network with fewer layers
            self.model = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(self.input_dim,)),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1)
            ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print(f"Built {architecture} neural network")
        return self.model
    
    def prepare_data(self, X, y, test_size=0.2, validation_split=0.1):
        """
        Prepare and scale data for training.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion for testing
            validation_split: Proportion of training data for validation
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Split data (don't shuffle for time series)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation data proportion
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            history: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=0
        )
        
        print(f"\nTraining Neural Network for {epochs} epochs...")
        print("="*50)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        print("="*50)
        print("Neural Network training complete!")
        
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        
        print("\n" + "="*50)
        print("Neural Network Test Set Evaluation")
        print("="*50)
        mae = metrics.get('mae', metrics.get('mean_absolute_error', 0))
        print(f"Loss (MSE):              {metrics['loss']:.2f}")
        print(f"Mean Absolute Error:     {mae:.2f}")
        print("="*50 + "\n")
        
        return metrics
    
    def save_model(self, filepath='models/saved/neural_network.keras'):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/saved/neural_network.keras'):
        """Load a previously saved model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def get_model_summary(self):
        """Print model architecture summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.summary()


if __name__ == "__main__":
    # Test the neural network
    print("Testing Neural Network Model...")
    print("="*50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 18
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(n_samples) * 0.5
    
    # Initialize model
    nn_model = NeuralNetworkModel(input_dim=n_features)
    
    # Build model
    nn_model.build_model(architecture='standard')
    nn_model.get_model_summary()
    
    # Prepare data
    X_train, X_test, y_train, y_test = nn_model.prepare_data(X, y)
    
    # Train model
    nn_model.train(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate
    nn_model.evaluate(X_test, y_test)
    
    # Make predictions
    predictions = nn_model.predict(X_test[:5])
    print("Sample predictions:", predictions)
    print("\nNeural Network test complete!")