import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
import numpy as np


from data import generate_synthetic_ohlcv_data, add_features

#tuning






# old model no tuning
def run_rolling_xgboost(df: pd.DataFrame, target_col: str = "return_1d",
                        window_size: int = 252, prediction_horizon: int = 1):
    """
    Run rolling-window XGBoost regression.
    """
    df_model = df.copy()
    features = df_model.drop(columns=[target_col]).columns.tolist()

    y_true = []
    y_pred = []
    pred_dates = []
    feature_importance = []

    for i in range(window_size, len(df_model) - prediction_horizon):
        train = df_model.iloc[i - window_size:i]
        test = df_model.iloc[i + prediction_horizon]

        X_train = train[features]
        y_train = train[target_col]

        X_test = df_model.iloc[[i]][features]
        y_test = test[target_col]

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, verbosity=0)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]
        y_pred.append(pred)
        y_true.append(y_test)
        pred_dates.append(df_model.index[i + prediction_horizon])
        feature_importance.append(model.feature_importances_)

    results_df = pd.DataFrame({
        "Date": pred_dates,
        "Actual": y_true,
        "Predicted": y_pred
    }).set_index("Date")

    avg_feature_importance = pd.DataFrame(feature_importance, columns=features).mean().sort_values(ascending=False)

    return results_df, avg_feature_importance



def evaluate_model(results_df: pd.DataFrame):
    y_true = results_df["Actual"]
    y_pred = results_df["Predicted"]

    # Basic Errors
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # Correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    # Directional Accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)

    # Predicted Sharpe Ratio (mean/std of predicted returns)
    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)
    sharpe_pred = mean_pred / std_pred if std_pred != 0 else np.nan

    # Output
    print("📊 Evaluation Metrics:")
    print(f"  ➤ MAE:                {mae:.6f}")
    print(f"  ➤ MSE:                {mse:.6f}")
    print(f"  ➤ R²:                 {r2:.6f}")
    print(f"  ➤ Explained Var:      {evs:.6f}")
    print(f"  ➤ MAPE (%):           {mape:.2f}")
    print(f"  ➤ Correlation:        {correlation:.4f}")
    print(f"  ➤ Direction Accuracy: {directional_accuracy:.2%}")
    print(f"  ➤ Predicted Sharpe:   {sharpe_pred:.4f}")
    print(f"  ➤ Pred Mean:          {mean_pred:.6f}")
    print(f"  ➤ Pred StdDev:        {std_pred:.6f}")


def plot_predictions(results_df: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    plt.plot(results_df.index, results_df["Actual"], label="Actual Return")
    plt.plot(results_df.index, results_df["Predicted"], label="Predicted Return", alpha=0.7)
    plt.title("Rolling XGBoost Predictions vs Actual")
    plt.ylabel("Return")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importances: pd.Series):
    plt.figure(figsize=(10, 6))
    importances.head(15).plot(kind="bar")
    plt.title("Average Feature Importance (XGBoost)")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = generate_synthetic_ohlcv_data(n_days=1500)
    df = add_features(df)

    results, importances = run_rolling_xgboost(df, target_col="target_return_1d", window_size=30)
    evaluate_model(results)
    plot_predictions(results)
    plot_feature_importance(importances)

'''
    # 3 day predict
    results, importances = run_rolling_xgboost(df, target_col="target_return_3d", window_size=30)
    evaluate_model(results)
    plot_predictions(results)
    plot_feature_importance(importances)'''

