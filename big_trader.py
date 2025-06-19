import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgb_tunign import run_pipeline

def generate_signals(predicted_returns: pd.Series, threshold: float = 0.005) -> pd.Series:
    """
    Generate trading signals (1 = Buy, -1 = Sell, 0 = Hold).
    """
    thr = abs(threshold)
    if thr == 0:
        return np.sign(predicted_returns).astype(int)
    signals = pd.Series(0, index=predicted_returns.index, dtype=int)
    signals[predicted_returns >= thr] = 1
    signals[predicted_returns <= -thr] = -1
    return signals

def apply_signals(signals: pd.Series, actual_returns: pd.Series,
                 initial_capital: float = 100000, position_fraction: float = 0.10,
                 use_position_sizing: bool = False, allow_shorting: bool = True,
                 transaction_cost: float = 0.0) -> pd.DataFrame:
    """
    Apply signals to compute capital evolution.
    """
    signals = signals.reindex(actual_returns.index)
    positions = signals.shift(1).fillna(0).astype(int)
    if not allow_shorting:
        positions = positions.clip(lower=0)

    #fees
    position_changes = positions.diff().fillna(0) != 0
    cost_penalty = position_changes.astype(float) * transaction_cost

    if use_position_sizing:
        raw_returns = positions * position_fraction * actual_returns
    else:
        raw_returns = positions * actual_returns

    strategy_returns = raw_returns - cost_penalty
    capital = initial_capital * (1 + strategy_returns.fillna(0)).cumprod()
    return pd.DataFrame({
        'position': positions,
        'signal': signals,
        'actual_return': actual_returns,
        'strategy_return': strategy_returns,
        'capital': capital
    })


def evaluate_strategy(strategy_df: pd.DataFrame) -> dict:
    """
    Evaluate strategy with performance metrics.
    """
    ret = strategy_df['strategy_return']
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std(ddof=0) != 0 else np.nan
    positions = strategy_df['position']
    trade_days = positions != 0
    if trade_days.any():
        correct = ((positions == 1) & (strategy_df['actual_return'] > 0)) | \
                  ((positions == -1) & (strategy_df['actual_return'] < 0))
        directional_acc = correct[trade_days].mean()
    else:
        directional_acc = np.nan

    num_trades = (strategy_df['position'].diff().fillna(0) != 0).sum()
    #max drawdown
    capital = strategy_df['capital']
    rolling_max = capital.cummax()
    drawdown = (capital - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return {
        'Sharpe Ratio (annualized)': sharpe,
        'Directional Accuracy': directional_acc,
        'Mean Strategy Return (daily)': ret.mean(),
        'Std Strategy Return (daily)': ret.std(),
        'Number of Trades': int(num_trades),
        'Max Drawdown': max_drawdown

    }


def run_trading_strategy(results_df: pd.DataFrame, threshold: float = 0.005,
                         initial_capital: float = 100000,
                         use_position_sizing: bool = False, position_fraction: float = 0.10,
                         allow_shorting: bool = True, transaction_cost: float=0.001) -> pd.DataFrame:
    """
    Backtest the trading strategy and generate metrics + plots.
    """
    if 'predicted_return' not in results_df.columns or 'actual_return' not in results_df.columns:
        raise KeyError("DataFrame must have 'predicted_return' and 'actual_return' columns.")

    signals = generate_signals(results_df['predicted_return'], threshold)
    strategy_df = apply_signals(signals, results_df['actual_return'],
                                initial_capital=initial_capital,
                                position_fraction=position_fraction,
                                use_position_sizing=use_position_sizing,
                                allow_shorting=allow_shorting, transaction_cost=transaction_cost)

    metrics = evaluate_strategy(strategy_df)
    print("\nTrading Strategy Performance:")
    print(f"Sharpe Ratio (annualized): {metrics['Sharpe Ratio (annualized)']:.2f}" if not pd.isna(metrics['Sharpe Ratio (annualized)']) else "Sharpe Ratio (annualized): N/A")
    print(f"Directional Accuracy: {metrics['Directional Accuracy'] * 100:.2f}%" if not pd.isna(metrics['Directional Accuracy']) else "Directional Accuracy: N/A")
    print(f"Mean Strategy Return (daily): {metrics['Mean Strategy Return (daily)'] * 100:.4f}%")
    print(f"Std Strategy Return (daily): {metrics['Std Strategy Return (daily)'] * 100:.4f}%")
    print(f"Number of Trades: {metrics['Number of Trades']}")
    print(f"Max Drawdown: {metrics['Max Drawdown'] * 100:.2f}%")

    plot_cumulative_returns(strategy_df, results_df, initial_capital)
    plot_trade_signals(strategy_df)
    plot_capital_curve(strategy_df)
    plot_drawdown(strategy_df)

    return strategy_df

def plot_cumulative_returns(strategy_df: pd.DataFrame, results_df: pd.DataFrame, initial_capital: float = 100000):
    strategy_cum = strategy_df['capital']
    actual_returns = results_df['actual_return'].reindex(strategy_df.index).fillna(0)
    buyhold_cum = initial_capital * (1 + actual_returns).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strategy_cum, label='Strategy', color='blue')
    ax.plot(buyhold_cum, label='Buy & Hold', color='orange')
    ax.set_title('Portfolio Value: Strategy vs Buy & Hold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='best')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_capital_curve(strategy_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy_df['capital'], label='Capital', color='purple')
    ax.set_title('Capital Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_trade_signals(strategy_df: pd.DataFrame):
    strategy_cum = strategy_df['capital']
    fig, ax = plt.subplots(figsize=(12, 6))
    buy = strategy_df[strategy_df['signal'] == 1]
    sell = strategy_df[strategy_df['signal'] == -1]
    hold = strategy_df[strategy_df['signal'] == 0]
    ax.plot(strategy_cum, color='black', alpha=0.3, label='Capital Curve')
    ax.scatter(buy.index, strategy_cum.loc[buy.index], marker='^', color='green', label='Buy', zorder=5)
    ax.scatter(sell.index, strategy_cum.loc[sell.index], marker='v', color='red', label='Sell', zorder=5)
    ax.scatter(hold.index, strategy_cum.loc[hold.index], marker='o', color='blue', label='Hold', alpha=0.4, zorder=5)
    ax.set_title('Trade Signals on Capital Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Capital ($)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drawdown(strategy_df: pd.DataFrame):
    capital = strategy_df['capital']
    rolling_max = capital.cummax()
    drawdown = (capital - rolling_max) / rolling_max

    plt.figure(figsize=(10, 5))
    plt.plot(drawdown, color='red')
    plt.title("Drawdown Over Time")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from data import generate_synthetic_ohlcv_data, add_features
    df = generate_synthetic_ohlcv_data(n_days=1500, trend_type='down' ) # OG 1500 # trend type 'up', 'down', or 'mixed'
    df = add_features(df)
    results_df = run_pipeline(df, target_col="target_return_5d", window_size=254) # 5 days , 254 window = very good
    results_df = results_df.rename(columns={"Predicted": "predicted_return", "Actual": "actual_return"})
    strategy_df = run_trading_strategy(results_df, threshold=0.005, initial_capital=100000,
                                       use_position_sizing=True, position_fraction=0.5, # %
                                       allow_shorting=True, transaction_cost=0.001) # 0.001 = 0.1%
