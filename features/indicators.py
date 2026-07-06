"""
Technical indicator computation.
Standalone function extracted from StockDataLoader.add_technical_indicators().
StockDataLoader now delegates to this function to avoid duplication.
"""
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 18 technical indicator columns to an OHLCV DataFrame.

    Does NOT mutate the input - returns a copy with indicators appended.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns
            and a DatetimeIndex.

    Returns:
        Copy of df with these columns added:
        MA_5, MA_10, MA_20, MA_50,
        EMA_12, EMA_26,
        MACD, Signal_Line,
        RSI,
        BB_Middle, BB_Upper, BB_Lower,
        Volume_MA_5, Volume_MA_20,
        Momentum_5, Momentum_10,
        Daily_Return, Volatility
    """
    df = df.copy()

    # --- Moving averages ---
    df['MA_5']  = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # --- Exponential moving averages ---
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # --- MACD ---
    df['MACD']        = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- RSI (14-day) ---
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands (20-day) ---
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std          = df['Close'].rolling(window=20).std()
    df['BB_Upper']  = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower']  = df['BB_Middle'] - (bb_std * 2)

    # --- Volume features ---
    df['Volume_MA_5']  = df['Volume'].rolling(window=5).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    # --- Price momentum ---
    df['Momentum_5']  = df['Close'].diff(5)
    df['Momentum_10'] = df['Close'].diff(10)

    # --- Returns and volatility ---
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility']   = df['Daily_Return'].rolling(window=20).std()

    return df


if __name__ == "__main__":
    from utils.data_loader import StockDataLoader
    loader = StockDataLoader("AAPL", "2022-01-01", "2024-01-01")
    loader.download_data()
    result = add_indicators(loader.raw_data)
    print(f"Columns added: {[c for c in result.columns if c not in loader.raw_data.columns]}")
    print(f"Output shape: {result.shape}")
    print("features/indicators.py: OK")
