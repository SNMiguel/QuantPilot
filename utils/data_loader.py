"""
Data loading and preprocessing utilities for stock prediction.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .sample_data import generate_sample_aapl_data
from features.indicators import add_indicators
import warnings
import logging

# Silence yfinance warnings and errors
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)


class StockDataLoader:
    """Handles downloading and preprocessing stock data."""
    
    def __init__(self, ticker="AAPL", start_date="2020-01-01", end_date=None):
        """
        Initialize the data loader.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format (default: today)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.raw_data = None
        
    def download_data(self, use_sample_if_fails=True):
        """Download stock data from Yahoo Finance."""
        print(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")
        
        try:
            import sys
            import io
            
            # Suppress yfinance error output
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            # Try downloading with different settings
            self.raw_data = yf.download(
                self.ticker, 
                start=self.start_date, 
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            
            # Restore stderr
            sys.stderr = old_stderr
            
            if self.raw_data.empty:
                raise ValueError(f"No data downloaded for {self.ticker}")
            
            # Handle multi-index columns if present
            if isinstance(self.raw_data.columns, pd.MultiIndex):
                self.raw_data.columns = self.raw_data.columns.get_level_values(0)
            
            self.data = self.raw_data.copy()
            print(f"Downloaded {len(self.data)} data points from Yahoo Finance.")
            return self.data
            
        except Exception as e:
            # Restore stderr if not already
            sys.stderr = old_stderr
            
            if use_sample_if_fails and self.ticker == "AAPL":
                print("WARN: Unable to fetch live data. Using sample AAPL data for demonstration...")
                self.raw_data = generate_sample_aapl_data(self.start_date, self.end_date)
                self.data = self.raw_data.copy()
                print(f"Generated {len(self.data)} sample data points.")
                return self.data
            else:
                raise ValueError(f"Could not download data for {self.ticker}.")
    def add_technical_indicators(self):
        """Add technical indicators as features. Delegates to features.indicators."""
        if self.data is None:
            raise ValueError("No data loaded. Call download_data() first.")

        self.data = add_indicators(self.data)
        print(f"Added {len(self.data.columns) - len(self.raw_data.columns)} technical indicators.")
        return self.data
    
    def prepare_features(self, target_column='Close', drop_na=True):
        """
        Prepare feature matrix and target variable.
        
        Args:
            target_column (str): Column to predict
            drop_na (bool): Whether to drop rows with NaN values
            
        Returns:
            tuple: (X, y) feature matrix and target variable
        """
        if self.data is None:
            raise ValueError("No data loaded. Call download_data() first.")
        
        # Drop NaN values created by technical indicators
        if drop_na:
            clean_data = self.data.dropna()
        else:
            clean_data = self.data.fillna(method='bfill').fillna(method='ffill')
        
        # Separate features and target
        y = clean_data[target_column].values
        
        # Select feature columns (everything except OHLCV)
        feature_columns = [col for col in clean_data.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        X = clean_data[feature_columns].values
        
        # Store feature names for later reference
        self.feature_names = feature_columns
        
        print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features.")
        return X, y, clean_data.index
    
    def get_basic_features(self):
        """Get simple time-based features (like your original project)."""
        if self.data is None:
            raise ValueError("No data loaded. Call download_data() first.")
        
        data_clean = self.data.dropna()
        data_clean['Day'] = np.arange(len(data_clean))
        X = data_clean[['Day']].values
        y = data_clean['Close'].values
        
        return X, y, data_clean.index


if __name__ == "__main__":
    # Test the data loader
    loader = StockDataLoader(ticker="AAPL", start_date="2020-01-01", end_date="2024-01-01")
    loader.download_data()
    loader.add_technical_indicators()
    X, y, dates = loader.prepare_features()
    
    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Feature names: {loader.feature_names}")