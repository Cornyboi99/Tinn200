import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgb_tunign import run_pipeline


def generate_signals(predicted_returns: pd.Series, threshold: float = 0.005) -> pd.Series:
    """
    Generate trading signals (1 for buy/long, -1 for sell/short, 0 for hold) from predicted returns using a threshold.

    If predicted_return >= threshold, signal = 1 (buy).
    If predicted_return <= -threshold, signal = -1 (sell).
    Otherwise, signal = 0 (hold).

    For threshold = 0, this is equivalent to taking the sign of the predicted returns.

    Parameters:
    predicted_returns (pd.Series): Series of predicted returns (as fractions, e.g. 0.01 for 1%).
    threshold (float): Minimum absolute return for taking a position (default 0.005).

    Returns:
    pd.Series: Series of signals (1, 0, -1) indexed as per predicted_returns.
    """
    thr = abs(threshold)
    if thr == 0:
        # If threshold is zero, use the sign of predicted returns (0 remains 0).
        return np.sign(predicted_returns).astype(int)
    # Initialize all signals to 0 (hold)
    signals = pd.Series(0, index=predicted_returns.index, dtype=int)
    # Assign buy (1) and sell (-1) signals based on threshold
    signals[predicted_returns >= thr] = 1
    signals[predicted_returns <= -thr] = -1
    return signals


def apply_signals(signals: pd.Series, actual_returns: pd.Series) -> pd.DataFrame:
    """
    Apply trading signals to actual returns to compute strategy positions and returns.

    The position on day t is the signal from day t-1 (shifted signals), to avoid using future information.

    Parameters:
    signals (pd.Series): Series of signals (1, -1, 0) indexed by date/time.
    actual_returns (pd.Series): Series of actual returns (fractional) indexed by date/time.

    Returns:
    pd.DataFrame: DataFrame with columns:
        'position': position held on each day (shifted signal),
        'strategy_return': return of strategy on each day (position * actual_return),
        'actual_return': the actual return of the asset on each day.
    """
    # Align signals with actual returns index
    signals = signals.reindex(actual_returns.index)
    # Lag the signals by one period to get positions (position at time t = signal from time t-1)
    positions = signals.shift(1).fillna(0).astype(int)
    # Calculate strategy returns: if long (1) then it's actual_return; if short (-1) it's -actual_return; if 0, then 0.
    strategy_returns = positions * actual_returns
    return pd.DataFrame({
        'position': positions,
        'strategy_return': strategy_returns,
        'actual_return': actual_returns,
        'signal': signals  # <-- Added this line to retain the original signal info

    })


def evaluate_strategy(strategy_df: pd.DataFrame) -> dict:
    """
    Calculate performance metrics for the strategy.

    Metrics:
    - Sharpe Ratio (annualized, 252 trading days, risk-free rate 0).
    - Directional Accuracy (ratio of correctly predicted directions when a trade is made).
    - Mean Strategy Return (daily average).
    - Std Strategy Return (daily standard deviation).

    Parameters:
    strategy_df (pd.DataFrame): DataFrame containing 'strategy_return', 'actual_return', and 'position' columns.

    Returns:
    dict: A dictionary of the performance metrics.
    """
    ret = strategy_df['strategy_return']
    # Sharpe Ratio (annualized): mean return / std dev of return * sqrt(252)
    if ret.std(ddof=0) == 0:
        sharpe = np.nan
    else:
        sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
    # Directional accuracy: only consider periods where a trade is made (position != 0)
    positions = strategy_df['position']
    trade_days = positions != 0
    if trade_days.any():
        # Correct trade if long and actual return > 0, or short and actual return < 0
        correct = ((positions == 1) & (strategy_df['actual_return'] > 0)) | \
                  ((positions == -1) & (strategy_df['actual_return'] < 0))
        directional_acc = correct[trade_days].mean()
    else:
        directional_acc = np.nan
    return {
        'Sharpe Ratio (annualized)': sharpe,
        'Directional Accuracy': directional_acc,
        'Mean Strategy Return (daily)': ret.mean(),
        'Std Strategy Return (daily)': ret.std(),
    }


def plot_cumulative_returns(strategy_df: pd.DataFrame):
    """
    Plot cumulative returns of the strategy vs a buy-and-hold approach.

    The buy-and-hold strategy assumes holding a long position in the asset for the entire period.

    Parameters:
    strategy_df (pd.DataFrame): DataFrame with 'strategy_return' and 'actual_return' columns.

    Returns:
    matplotlib.figure.Figure: Figure object of the generated plot.
    """
    # Compute cumulative return paths (growth of $1 initial investment)
    print(strategy_df)
    strategy_cum = (1 + strategy_df['strategy_return'].fillna(0)).cumprod()
    buyhold_cum = (1 + strategy_df['actual_return'].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strategy_cum, label='Strategy')
    ax.plot(buyhold_cum, label='Buy & Hold')
    ax.set_title('Cumulative Returns: Strategy vs Buy & Hold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Growth of $1)')
    ax.legend(loc='best')
    ax.grid(True)
    return fig

def plot_trade_signals(strategy_df: pd.DataFrame):
    strategy_cum = (1 + strategy_df['strategy_return'].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    buy = strategy_df[strategy_df['signal'] == 1]
    sell = strategy_df[strategy_df['signal'] == -1]
    hold = strategy_df[strategy_df['signal'] == 0]

    ax.plot(strategy_cum, color='black', alpha=0.3, label='Cumulative Strategy Return')
    ax.scatter(buy.index, strategy_cum.loc[buy.index], marker='^', color='green', label='Buy', zorder=5)
    ax.scatter(sell.index, strategy_cum.loc[sell.index], marker='v', color='red', label='Sell', zorder=5)
    ax.scatter(hold.index, strategy_cum.loc[hold.index], marker='o', color='blue', label='Hold', alpha=0.4, zorder=5)

    ax.set_title('Trade Signals on Cumulative Strategy Return')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def run_trading_strategy(results_df: pd.DataFrame, threshold: float = 0.005):
    """
    Run the trading strategy:
      1. Generate signals from predicted returns using the given threshold.
      2. Determine positions and strategy returns by shifting signals.
      3. Evaluate performance metrics.
      4. Plot cumulative returns vs buy-and-hold.

    Prints the performance metrics and displays the cumulative returns plot.

    Parameters:
    results_df (pd.DataFrame): DataFrame with columns 'predicted_return' and 'actual_return'.
    threshold (float): Threshold for signal generation (default 0.005).

    Returns:
    pd.DataFrame: DataFrame with 'position', 'strategy_return', and 'actual_return' for further analysis.
    """
    # Ensure required columns are present
    if 'predicted_return' not in results_df.columns or 'actual_return' not in results_df.columns:
        raise KeyError("DataFrame must have 'predicted_return' and 'actual_return' columns.")
    # Step 1: Generate signals
    signals = generate_signals(results_df['predicted_return'], threshold)
    # Step 2: Apply signals to compute positions and strategy returns
    strategy_df = apply_signals(signals, results_df['actual_return'])
    print(strategy_df.head())
    # Step 3: Evaluate strategy performance
    metrics = evaluate_strategy(strategy_df)
    # Print metrics
    print("Trading Strategy Performance:")
    # Sharpe Ratio
    if pd.isna(metrics['Sharpe Ratio (annualized)']):
        print("Sharpe Ratio (annualized): N/A (no trades or zero variance)")
    else:
        print(f"Sharpe Ratio (annualized): {metrics['Sharpe Ratio (annualized)']:.2f}")
    # Directional Accuracy
    if pd.isna(metrics['Directional Accuracy']):
        print("Directional Accuracy: N/A (no trades executed)")
    else:
        print(f"Directional Accuracy: {metrics['Directional Accuracy'] * 100:.2f}%")
    # Mean and Std of strategy returns
    print(f"Mean Strategy Return (daily): {metrics['Mean Strategy Return (daily)'] * 100:.4f}%")
    print(f"Std Strategy Return (daily): {metrics['Std Strategy Return (daily)'] * 100:.4f}%")
    # Step 4: Plot cumulative returns
    fig = plot_cumulative_returns(strategy_df)
    plt.show()
    plot_trade_signals(strategy_df)

    return strategy_df


# Example usage / self-test (executes only if run as a script):
if __name__ == "__main__":
    # Step 1: Run XGBoost pipeline
    results_df = run_pipeline(target_col="target_return_5d", window_size=60) #5 days , WS = 254 very good

    # Step 2: Prepare for classification-style trading
    results_df = results_df.rename(columns={"Predicted": "predicted_return", "Actual": "actual_return"})

    # Step 3: Apply your trading logic
    run_trading_strategy(results_df, threshold=0.005) #0.005 = 5%



    '''# Generate a small synthetic dataset for demonstration
    dates = pd.date_range("2022-01-01", periods=15, freq='D')
    rng = np.random.default_rng(42)
    # Simulate actual returns (e.g., daily returns around 0 with some volatility)
    actual_rets = pd.Series(rng.normal(0, 0.01, size=len(dates)), index=dates)
    # Simulate predicted returns (use previous actual return + noise to mimic prediction)
    predicted_rets = actual_rets.shift(1).fillna(0) + rng.normal(0, 0.005, size=len(dates))
    # Prepare results DataFrame
    demo_df = pd.DataFrame({'predicted_return': predicted_rets, 'actual_return': actual_rets})
    # Run the trading strategy analysis on the demo data
    run_trading_strategy(demo_df, threshold=0.005)
'''