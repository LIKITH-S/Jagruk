import pandas as pd
import numpy as np

def calculate_delta(df, column, periods=1, target_name=None):
    """
    Calculates the absolute difference from a previous period (rate of change).
    """
    df_out = df.copy()
    if column not in df_out.columns:
        return df_out
        
    feature_name = target_name if target_name else f"{column}_delta_{periods}"
    df_out[feature_name] = df_out[column].diff(periods=periods)
    return df_out

def calculate_slope(series, window=3):
    """
    Calculates a rolling linear regression slope over a given window.
    """
    def slope(y):
        if len(y) < 2 or np.isnan(y).any():
            return 0.0
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    
    return series.rolling(window=window, min_periods=2).apply(slope, raw=True)

def add_trend_features(df, column, window=3, target_name=None):
    """
    Adds rolling slope features to capture the underlying trend.
    """
    df_out = df.copy()
    if column not in df_out.columns:
        return df_out
        
    feature_name = target_name if target_name else f"{column}_trend_{window}"
    df_out[feature_name] = calculate_slope(df_out[column], window=window)
    return df_out
