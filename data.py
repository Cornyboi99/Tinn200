import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 2 different dataset to try, one with log rise of stock, one with extreme volatility
# and other with huge drops

def generate_synthetic_ohlcv_data(n_days=1000, start_price=100.0, seed=42, trend_type="mixed"):
    """
    Generates synthetic OHLCV data with optional trend types: 'up', 'down', or 'mixed'.

    Parameters:
        n_days (int): Number of days of data
        start_price (float): Starting price
        seed (int): Random seed for reproducibility
        trend_type (str): One of 'up', 'down', or 'mixed'

    Returns:
        pd.DataFrame: Synthetic OHLCV data
    """
    np.random.seed(seed)
    start_date = datetime(2019, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    if trend_type == "up":
        log_returns = np.random.normal(loc=0.0005, scale=0.02, size=n_days)


    elif trend_type == "down":
        half = n_days // 2
        # First half: small uptrend
        uptrend = np.random.normal(loc=0.0005, scale=0.01, size=half)
        # Second half: stronger downtrend
        downtrend = np.random.normal(loc=-0.003, scale=0.015, size=n_days - half)
        log_returns = np.concatenate([uptrend, downtrend])
        # Add shocks only in the second half to simulate crashes
        shock_days = np.random.choice(np.arange(half, n_days), size=10, replace=False)
        log_returns[shock_days] += np.random.choice([-0.05, -0.1, -0.15], size=10)

    elif trend_type == "mixed":
        log_returns = np.random.normal(0.0003, 0.015, size=n_days)
        shock_days = np.random.choice(n_days, size=10, replace=False)
        log_returns[shock_days] += np.random.choice([-0.2, 0.2], size=10)

    else:
        raise ValueError("Invalid trend_type. Choose from 'up', 'down', or 'mixed'.")

    prices = start_price * np.exp(np.cumsum(log_returns))
    close = prices
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = close * (1 + np.random.uniform(0.001, 0.03, size=n_days))
    low = close * (1 - np.random.uniform(0.001, 0.03, size=n_days))
    volume = np.random.randint(1e5, 5e6, size=n_days)

    df = pd.DataFrame({
        "timestamp": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })
    df.set_index("timestamp", inplace=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import ta

    df_feat = df.copy()

    # === Target & Lag Returns ===
    # & Lag Returns
    df_feat['return_1'] = df_feat['Close'].pct_change(1)
    df_feat['return_2'] = df_feat['Close'].pct_change(2)
        #days to predict for % change
    df_feat['target_return_1d'] = df_feat['return_1'].shift(-1)  # 1-day forward return
    df_feat['target_return_3d'] = (df_feat['Close'].shift(-3) - df_feat['Close']) / df_feat['Close']
    df_feat['target_return_5d'] = (df_feat['Close'].shift(-5) - df_feat['Close']) / df_feat['Close']

    df_feat['lag1'] = df_feat['return_1'].shift(1)

    # === Rolling Stats ===
    df_feat['rolling_mean_5'] = df_feat['Close'].rolling(window=5).mean()
    df_feat['rolling_std_5'] = df_feat['Close'].rolling(window=5).std()

    #New rolling # just chaneg window for longer period
    df_feat['rolling_mean_10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['rolling_std_10'] = df_feat['Close'].rolling(window=10).std()

    df_feat['rolling_median_10'] = df_feat['Close'].rolling(window=10).median()

    for window in [10]: #drop some if high correlation, not chosen yet 5,10,15,20
        df_feat[f'rolling_min_{window}'] = df_feat['Close'].rolling(window=window).min()
        df_feat[f'rolling_max_{window}'] = df_feat['Close'].rolling(window=window).max()
        df_feat[f'price_vs_min_{window}'] = df_feat['Close'] / df_feat[f'rolling_min_{window}']
        df_feat[f'price_vs_max_{window}'] = df_feat['Close'] / df_feat[f'rolling_max_{window}']

    # Price relative to recent mean or max
    df_feat['price_vs_mean_5'] = df_feat['Close'] / df_feat['rolling_mean_5']

    # === Technical Indicators ===
    df_feat['rsi_14'] = ta.momentum.RSIIndicator(close=df_feat['Close'], window=14, fillna=True).rsi()

    macd = ta.trend.MACD(close=df_feat['Close'], fillna=True)
    df_feat['macd'] = macd.macd()
    df_feat['macd_signal'] = macd.macd_signal()

    # === Log Return ===
    df_feat['log_return'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1)).replace([np.inf, -np.inf], np.nan)

    # === Volatility ===
    df_feat['volatility_5d'] = df_feat['return_1'].rolling(window=5).std()

    # === ADX ===
    df_feat['adx'] = ta.trend.ADXIndicator(high=df_feat['High'], low=df_feat['Low'], close=df_feat['Close'], fillna=True).adx()

    # === Volume Z-Score ===
    df_feat['volume_z'] = (df_feat['Volume'] - df_feat['Volume'].rolling(window=20).mean()) / df_feat['Volume'].rolling(window=20).std()

    # Final cleanup
    df_feat.dropna(inplace=True)

    return df_feat

import matplotlib.pyplot as plt

def plot_ohlcv(df: pd.DataFrame, title="Synthetic OHLCV Data"):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close Price")
    plt.fill_between(df.index, df["Low"], df["High"], color='lightgray', alpha=0.4, label="High-Low Range")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pca_scree(df: pd.DataFrame, n_components: int = 10):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = df.drop(columns=[col for col in df.columns if col.startswith("target_return") or col.startswith("return_1d")])
    X = X.dropna()

    # Standardize features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, X.shape[1]))
    pca.fit(X_scaled)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
    plt.title("Scree Plot: Explained Variance by Principal Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_correlation(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select only numeric features
    numeric_df = df.select_dtypes(include=[float, int]).copy()
    numeric_df = numeric_df.drop(columns=[col for col in numeric_df.columns if col.startswith("target_return") or col.startswith("return_1d")], errors='ignore')

    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = generate_synthetic_ohlcv_data(n_days=1500, trend_type='mixed')
    df = add_features(df)
    plot_ohlcv(df)
    print(df.head())
    print(df.columns.tolist())

    plot_feature_correlation(df)
    plot_pca_scree(df)