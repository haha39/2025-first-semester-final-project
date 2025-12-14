import yfinance as yf
import pandas as pd
import numpy as np

def download_stock_data(stock_id, start_date='2020-01-01', end_date='2024-12-31'):
    """
    下載股票資料

    Args:
        stock_id: 股票代碼 (例如 '2330.TW')
        start_date: 開始日期
        end_date: 結束日期

    Returns:
        DataFrame: 股價資料
    """
    print(f"正在下載 {stock_id} 的資料...")
    data = yf.download(stock_id, start=start_date, end=end_date, progress=False)

    # 處理多層索引問題（當只下載一支股票時，yfinance 可能返回多層索引）
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    print(f"下載完成！共 {len(data)} 筆資料")
    return data

def engineer_features(data):
    """
    建立完整的技術特徵

    Args:
        data: 原始股價 DataFrame

    Returns:
        DataFrame: 包含所有技術特徵的 DataFrame
    """
    df = data.copy()

    # 1. RSI (標準化至 0-1)
    df['RSI'] = calculate_rsi(df, period=14) / 100.0

    # 2. SMA 與乖離率
    df['SMA_20'] = calculate_sma(df, period=20)
    df['SMA_Deviation'] = (df['Close'] - df['SMA_20']) / df['SMA_20']

    # 3. ATR (波動率指標)
    df['ATR'] = calculate_atr(df, period=14)
    df['ATR_Normalized'] = df['ATR'] / df['Close']  # 標準化

    # 4. 對數收益率
    log_returns = calculate_log_returns(df, periods=[5, 10, 20])
    df = pd.concat([df, log_returns], axis=1)

    # 5. 成交量變化率
    df['Volume_Change'] = df['Volume'].pct_change(5)

    # 6. 價格動能 (Momentum)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Momentum_20'] = df['Close'] - df['Close'].shift(20)

    # 7. 處理無限值和超大值（替換為 NaN，後續會被 dropna 移除）
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def create_labels(data, threshold=0.004, hold_threshold=0.002):
    """
    建立預測標籤：預測明日的操作策略（三分類）
    考慮交易成本，設定門檻為 0.4% (手續費0.1425% + 證交稅0.3%)

    Args:
        data: DataFrame
        threshold: 買入門檻 (預設 0.4%)
        hold_threshold: 持有門檻 (預設 0.2%)

    Returns:
        Series: 標籤 (0=賣出/觀望, 1=持有, 2=買入)
    """
    future_close = data['Close'].shift(-1)
    future_open = data['Open'].shift(-1)

    # 計算預期收益率
    expected_return = (future_close - future_open) / future_open

    # 三分類標籤
    # 2: 預期收益 > threshold (買入)
    # 1: -hold_threshold <= 預期收益 <= threshold (持有)
    # 0: 預期收益 < -hold_threshold (賣出/觀望)
    labels = pd.Series(1, index=data.index)  # 預設為持有
    labels[expected_return > threshold] = 2  # 買入
    labels[expected_return < -hold_threshold] = 0  # 賣出/觀望

    return labels

def build_dataset(ticker, config):
    """
    建立訓練與測試資料集

    Args:
        ticker: 股票代碼
        config: 包含 start_date, end_date, train_end 的設定字典

    Returns:
        tuple: (X_train, y_train, X_test, y_test, test_meta_df)
    """
    # 下載資料
    data = download_stock_data(ticker, config['start_date'], config['end_date'])

    # 建立特徵與標籤
    data = engineer_features(data)
    data['Target'] = create_labels(data)

    # 移除 NaN
    data = data.dropna()

    # 切分訓練集與測試集
    train_data = data.loc[:config['train_end']]
    test_data = data.loc[config['train_end']:]

    X_train = train_data.drop(columns=['Target'])
    y_train = train_data['Target']
    X_test = test_data.drop(columns=['Target'])
    y_test = test_data['Target']

    # 建立 test_meta_df
    test_meta_df = test_data[['Date', 'Open', 'Close']]

    return X_train, y_train, X_test, y_test, test_meta_df