import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data import  generate_synthetic_ohlcv_data, add_features, plot_ohlcv

# not fully tested

API_KEY = "YOUR_POLYGON_API_KEY"
BASE_URL = "https://api.polygon.io"

def get_polygon_data(ticker: str, start_date: str, end_date: str,
                     timespan: str = "day", multiplier: int = 1) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Polygon.io for a given ticker/date range.
    """
    url = (f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/"
           f"{start_date}/{end_date}?apiKey={API_KEY}")
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Polygon API error {resp.status_code}: {resp.text}")
    results = resp.json().get("results", [])
    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'}, inplace=True)
    return df[['Open','High','Low','Close','Volume']]

def load_data(source: str = 'synthetic', **kwargs) -> pd.DataFrame:
    """
    Choose data source: 'synthetic' or 'polygon'.
    """
    if source == 'polygon':
        return get_polygon_data(kwargs['ticker'], kwargs['start_date'], kwargs['end_date'],
                                 timespan=kwargs.get('timespan', 'day'),
                                 multiplier=kwargs.get('multiplier', 1))
    elif source == 'synthetic':
        return generate_synthetic_ohlcv_data(n_days=kwargs.get('n_days',1000),
                                             start_price=kwargs.get('start_price',100.0),
                                             seed=kwargs.get('seed',42),
                                             trend_type=kwargs.get('trend_type','mixed'))
    else:
        raise ValueError("source must be 'synthetic' or 'polygon'")

# Example usage
if __name__ == "__main__":
    # Synthetic example:
    df = load_data(source='synthetic', n_days=1500, trend_type='mixed')

    # Polygon example:
    # df = load_data(source='polygon', ticker='AAPL',
    #                start_date='2022-01-01', end_date='2023-01-01',
    #                timespan='day', multiplier=1)

    df_feat = add_features(df)
    plot_ohlcv(df, title="OHLCV Data")
