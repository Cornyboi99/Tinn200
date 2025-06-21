from data import add_features
from XGB_regressor import run_rolling_xgboost
import pandas as pd
from api_data import get_polygon_data
from Tradeing_XGB import generate_signals, apply_signals

# == Example of Live trading, not complete BTW ==

# Settings
window_days = 300
TARGET = "target_return_5d"
INITIAL_CAPITAL = 100000


# Fetch fresh data
# Parameters
ticker = "AAPL"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
start_date = (pd.Timestamp.today() - pd.Timedelta(days=window_days + 10)).strftime("%Y-%m-%d")

# Fetch live data
df_live = get_polygon_data(ticker, start_date, end_date)

df_feat = add_features(df_live)

# Slice the latest rolling window
df_window = df_feat[-(window_days + 5):].copy()

# train or with a pretrained model etc

results, _ = run_rolling_xgboost(
    df_window,
    target_col=TARGET,
    window_size=window_days,
    tuned_params=None,  # Optionally load your best_params from file
    use_pca=False
)

#  Get last predicted return
latest_row = results.iloc[-1]
predicted_return = latest_row["Predicted"]

# Generate trading signal
signal = generate_signals(pd.Series([predicted_return]), threshold=0.005)[0]

print(f"📆 Date: {latest_row.name.date()} | Predicted: {predicted_return:.4f} | Signal: {signal}")

# simulate today's return and update capital
actual_return = latest_row["Actual"]
strategy_df = apply_signals(
    signals=pd.Series([signal], index=[latest_row.name]),
    actual_returns=pd.Series([actual_return], index=[latest_row.name]),
    initial_capital=INITIAL_CAPITAL,
    position_fraction=0.5,
    use_position_sizing=True,
    allow_shorting=True,
    transaction_cost=0.001
)

print(strategy_df[['capital', 'strategy_return']])
