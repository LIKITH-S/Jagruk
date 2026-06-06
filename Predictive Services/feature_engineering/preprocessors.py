import pandas as pd

class DataPreprocessor:
    """Handles basic time-series preprocessing steps."""
    
    @staticmethod
    def normalize_timestamp(df, time_column='date'):
        """Converts timestamp to datetime and sets it as index."""
        df_out = df.copy()
        if time_column in df_out.columns:
            df_out[time_column] = pd.to_datetime(df_out[time_column])
            df_out = df_out.sort_values(by=time_column)
            df_out = df_out.set_index(time_column)
        return df_out

    @staticmethod
    def handle_missing_values(df, strategy='ffill'):
        """Handles missing values using forward fill, backward fill, or drop."""
        if strategy == 'ffill':
            return df.ffill()
        elif strategy == 'bfill':
            return df.bfill()
        elif strategy == 'drop':
            return df.dropna()
        return df

    @staticmethod
    def interpolate_values(df, method='linear'):
        """Interpolates missing numerical values."""
        # Ensure we only interpolate numeric columns to avoid warnings
        numeric_cols = df.select_dtypes(include='number').columns
        df_out = df.copy()
        df_out[numeric_cols] = df_out[numeric_cols].interpolate(method=method)
        return df_out

    @staticmethod
    def resample_data(df, freq='1h', agg_methods=None):
        """Resamples time-series data to a specific frequency."""
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError("DataFrame index must be a datetime object for resampling.")
            
        if agg_methods is None:
            # Default aggregation is mean
            return df.resample(freq).mean()
        return df.resample(freq).agg(agg_methods)
