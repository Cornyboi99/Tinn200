from data import add_features, generate_synthetic_ohlcv_data
from api_data import load_data
from xgb_tunign import run_pipeline  # your XGBoost model pipeline
from big_trader import run_trading_strategy  # trading strategy logic

def main():
    # === Configuration ===
    DATA_SOURCE = "synthetic"  # or "polygon"
    PREDICT_HORIZON = "target_return_5d"
    WINDOW_SIZE = 254
    TREND_TYPE = "mixed"
    DAYS  = 1500

    # === Load data ===
    if DATA_SOURCE == "polygon":
        df = load_data(
            source='polygon',
            ticker='AAPL',
            start_date='2022-01-01',
            end_date='2023-01-01',
            timespan='day',
            multiplier=1
        )
    else:
        df = load_data(
            source='synthetic',
            n_days=DAYS,
            trend_type=TREND_TYPE,
            start_price=100.0,
            seed=42
        )

    # === Feature Engineering ===
    df_feat = add_features(df)

    # === Model Pipeline ===
    results_df = run_pipeline(df_feat, target_col=PREDICT_HORIZON, window_size=WINDOW_SIZE)

    # === Run Backtest ===
    results_df = results_df.rename(columns={"Predicted": "predicted_return", "Actual": "actual_return"})
    strategy_df = run_trading_strategy(
        results_df,
        threshold=0.005,
        initial_capital=100000,
        use_position_sizing=True,
        position_fraction=0.50,
        allow_shorting=True,
        transaction_cost=0.001  # 0.1%
    )

if __name__ == "__main__":
    main()
