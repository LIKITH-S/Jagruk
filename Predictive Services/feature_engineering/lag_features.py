import pandas as pd

def add_lag_features(df, column, lags):
    """
    Calculates lag features (historical values) for a specified column.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Name of the column to lag.
        lags (list): List of integer periods to lag.
        
    Returns:
        pd.DataFrame: DataFrame with lag features added.
    """
    df_out = df.copy()
    if column not in df_out.columns:
        return df_out
        
    for lag in lags:
        feature_name = f"{column}_lag_{lag}"
        df_out[feature_name] = df_out[column].shift(lag)
        
    return df_out
