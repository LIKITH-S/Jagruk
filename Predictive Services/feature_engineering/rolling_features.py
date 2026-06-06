import pandas as pd

def add_rolling_sum(df, column, windows, suffix='h'):
    """
    Calculates rolling sum over specified windows (e.g., rainfall_3h).
    Handles both datetime indexed data (time-based windows) and 
    integer indexed data (row-based windows).
    """
    df_out = df.copy()
    if column not in df_out.columns:
        return df_out
        
    for w in windows:
        feature_name = f"{column}_{w}{suffix}"
        
        if pd.api.types.is_datetime64_any_dtype(df_out.index):
            window_str = f"{w}{suffix}"
            df_out[feature_name] = df_out[column].rolling(window_str, min_periods=1).sum()
        else:
            df_out[feature_name] = df_out[column].rolling(window=w, min_periods=1).sum()
            
    return df_out

def add_rolling_average(df, column, windows, suffix='h'):
    """Calculates rolling average over specified windows."""
    df_out = df.copy()
    if column not in df_out.columns:
        return df_out
        
    for w in windows:
        feature_name = f"{column}_avg_{w}{suffix}"
        
        if pd.api.types.is_datetime64_any_dtype(df_out.index):
            window_str = f"{w}{suffix}"
            df_out[feature_name] = df_out[column].rolling(window_str, min_periods=1).mean()
        else:
            df_out[feature_name] = df_out[column].rolling(window=w, min_periods=1).mean()
            
    return df_out
