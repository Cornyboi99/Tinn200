import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.decomposition import PCA

from data import generate_synthetic_ohlcv_data, add_features

# Prepare static training data for tuning
def prepare_static_training_data(df, target_col, train_size=0.7):
    df_train = df.iloc[:int(len(df)*train_size)].copy()
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]
    return X, y

# Define the objective function for Bayesian optimization
X_train, y_train = None, None  # global placeholder

def xgb_cv_eval(learning_rate, max_depth, subsample, colsample_bytree):
    max_depth = int(round(max_depth))
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        verbosity=0
    )
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    return scores.mean()

# Updated model using tuned parameters

def run_rolling_xgboost(df: pd.DataFrame, target_col: str = "return_1d",
                        window_size: int = 252, prediction_horizon: int = 1,
                        tuned_params: dict = None, use_pca: bool = False, n_components: int = 5):
    df_model = df.copy()
    # Drop target and all other columns starting with "target_return"
    features = df_model.drop(
        columns=[col for col in df_model.columns if col.startswith("target_return")]).columns.tolist()

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

        if use_pca:
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            verbosity=0,
            random_state=42,
            **(tuned_params or {})
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]
        y_pred.append(pred)
        y_true.append(y_test)
        pred_dates.append(df_model.index[i + prediction_horizon])
        if not use_pca:
            feature_importance.append(model.feature_importances_)

    results_df = pd.DataFrame({
        "Date": pred_dates,
        "Actual": y_true,
        "Predicted": y_pred
    }).set_index("Date")

    if not use_pca:
        avg_feature_importance = pd.DataFrame(feature_importance, columns=features).mean().sort_values(ascending=False)
    else:
        avg_feature_importance = pd.Series(dtype=float)  # empty if PCA used

    return results_df, avg_feature_importance


def evaluate_model(results_df: pd.DataFrame):
    y_true = results_df["Actual"]
    y_pred = results_df["Predicted"]

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)
    sharpe_pred = mean_pred / std_pred if std_pred != 0 else np.nan

    print("\U0001F4CA Evaluation Metrics:")
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


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_predictions(results_df: pd.DataFrame, initial_price: float = 100):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # 1. Plot Actual Returns Only
    axs[0].plot(results_df.index, results_df["Actual"], label="Actual Return", color="blue")
    axs[0].set_title("Actual Returns Only")
    axs[0].set_ylabel("Return")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Actual vs Predicted Returns
    axs[1].plot(results_df.index, results_df["Actual"], label="Actual Return", color="blue")
    axs[1].plot(results_df.index, results_df["Predicted"], label="Predicted Return", color="orange", alpha=0.7)
    axs[1].set_title("Rolling XGBoost Predictions vs Actual")
    axs[1].set_ylabel("Return")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances: pd.Series):
    if not importances.empty:
        plt.figure(figsize=(10, 6))
        importances.head(15).plot(kind="bar")
        plt.title("Average Feature Importance (XGBoost)")
        plt.ylabel("Importance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print('empty feature')


def run_pipeline(df, target_col="target_return_5d", window_size=254):

    #target_col = "target_return_5d"
    #window_size = 254

    global X_train, y_train
    X_train, y_train = prepare_static_training_data(df, target_col=target_col)

    pbounds = {
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    }

    optimizer = BayesianOptimization(
        f=xgb_cv_eval,
        pbounds=pbounds,
        random_state=42,
    )
    optimizer.maximize(init_points=5, n_iter=20)

    best_params = optimizer.max['params']
    best_params['max_depth'] = int(round(best_params['max_depth']))
    print("Best parameters:", best_params)

    results, importances = run_rolling_xgboost(
        df,
        target_col=target_col,
        window_size=window_size,
        tuned_params=best_params,
        use_pca=False,
        n_components=3
    )

    evaluate_model(results)
    plot_predictions(results)
    plot_feature_importance(importances)

    return results


if __name__ == "__main__":
    target_col = "target_return_5d"
    window_size = 254
    run_pipeline(target_col="target_return_5d", window_size=254)


