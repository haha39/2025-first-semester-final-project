import yfinance as yf
import pandas as pd
import numpy as np

def download_stock_data(stock_id, start_date='2020-01-01', end_date='2024-12-31'):
    """
    Download stock data.

    Args:
        stock_id: Stock ticker symbol (e.g., '2330.TW').
        start_date: Start date for the data.
        end_date: End date for the data.

    Returns:
        DataFrame: Stock price data.
    """
    print(f"Downloading data for {stock_id}...")
    data = yf.download(stock_id, start=start_date, end=end_date, progress=False)

    # Handle multi-level index issues (occurs when downloading a single stock with yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    print(f"Download complete! Total {len(data)} records.")
    return data

def engineer_features(data):
    """
    Generate comprehensive technical features.

    Args:
        data: Raw stock price DataFrame.

    Returns:
        DataFrame: DataFrame containing all technical features.
    """
    df = data.copy()

    # 1. RSI (normalized to 0-1)
    df['RSI'] = calculate_rsi(df, period=14) / 100.0

    # 2. SMA and deviation
    df['SMA_20'] = calculate_sma(df, period=20)
    df['SMA_Deviation'] = (df['Close'] - df['SMA_20']) / df['SMA_20']

    # 3. ATR (volatility indicator)
    df['ATR'] = calculate_atr(df, period=14)
    df['ATR_Normalized'] = df['ATR'] / df['Close']  # Normalized

    # 4. Log returns
    log_returns = calculate_log_returns(df, periods=[5, 10, 20])
    df = pd.concat([df, log_returns], axis=1)

    # 5. Volume change rate
    df['Volume_Change'] = df['Volume'].pct_change(5)

    # 6. Price momentum (Momentum)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Momentum_20'] = df['Close'] - df['Close'].shift(20)

    # 7. Handle infinite and large values (replace with NaN, which will be removed by subsequent dropna)
    df = df.replace([np.inf, -np.inf], np.nan)

def create_labels(data, threshold=0.004, hold_threshold=0.002):
    """
    Create prediction labels: Predict the next day's trading strategy (three classes).
    Considers transaction costs, with a threshold set at 0.4% (0.1425% fee + 0.3% tax).

    Args:
        data: DataFrame containing stock data.
        threshold: Buy threshold (default 0.4%).
        hold_threshold: Hold threshold (default 0.2%).

    Returns:
        Series: Labels (0 = sell/observe, 1 = hold, 2 = buy).
    """
    future_close = data['Close'].shift(-1)
    future_open = data['Open'].shift(-1)

    # Calculate expected return
    expected_return = (future_close - future_open) / future_open

    # Three-class labels
    # 2: Expected return > threshold (buy)
    # 1: -hold_threshold <= Expected return <= threshold (hold)
    # 0: Expected return < -hold_threshold (sell/observe)
    labels = pd.Series(1, index=data.index)  # Default to hold
    labels[expected_return > threshold] = 2  # Buy
    labels[expected_return < -hold_threshold] = 0  # Sell/observe

    return labels

def build_dataset(ticker, config):
    """
    Build training and testing datasets.

    Args:
        ticker: Stock ticker symbol.
        config: Dictionary containing start_date, end_date, and train_end.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, test_meta_df).
    """
    # Download data
    data = download_stock_data(ticker, config['start_date'], config['end_date'])

    # Generate features and labels
    data = engineer_features(data)
    data['Target'] = create_labels(data)

    # Remove NaN values
    data = data.dropna()

    # Split into training and testing datasets
    train_data = data.loc[:config['train_end']]
    test_data = data.loc[config['train_end']:]

    X_train = train_data.drop(columns=['Target'])
    y_train = train_data['Target']
    X_test = test_data.drop(columns=['Target'])
    y_test = test_data['Target']

    # Create test_meta_df
    test_meta_df = test_data[['Date', 'Open', 'Close']]

    return X_train, y_train, X_test, y_test, test_meta_df