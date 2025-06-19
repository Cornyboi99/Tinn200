# 🧠 AI Hedge Fund – Algorithmic Trading with XGBoost

This project showcases a simplified AI-driven hedge fund that uses machine learning (XGBoost) to predict asset returns and generate trading signals. It supports both **synthetic data** and **real market data** via the Polygon.io API.

Made by Cornelius Birkelid Lekman and Taofik Muhriz 
---

## 🔧 Features

- 📊 **Synthetic OHLCV data generator** with uptrend, downtrend, or mixed markets
- 🧪 **Feature engineering** including:
  - Technical indicators (RSI, MACD, ADX)
  - Rolling statistics
  - Log returns and volatility
- 🧠 **XGBoost regression model** with rolling-window backtesting
- 🤖 **Trading agent** with:
  - Threshold-based decision rules
  - Long/short trading support
  - Position sizing
  - Transaction cost modeling
- 📈 **Evaluation metrics**:
  - Sharpe Ratio
  - Directional Accuracy
  - Max Drawdown
  - Capital curves
- 🌐 **Live data support** (optional) via the Polygon.io API

---

## 📁 File Overview

```bash
project/
├── main.py                    # Main entry point (data + model + strategy)
├── data.py                    # Synthetic generator + Polygon API access
├── xgb_tunign.py              # Model training, prediction, cross-validation
├── classification_trading.py # Trading logic, metrics, backtest, plots
