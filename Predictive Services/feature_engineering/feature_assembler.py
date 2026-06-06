import pandas as pd
from .preprocessors import DataPreprocessor
from .scaling import FeatureScaler
from .rolling_features import add_rolling_sum, add_rolling_average
from .lag_features import add_lag_features
from .trend_features import calculate_delta, add_trend_features

class FeatureAssembler:
    """
    Orchestrates the preprocessing and feature engineering pipeline.
    Transforms raw datasets into fully prepared features for ML models.
    """
    def __init__(self, is_training=True, scaler_method='standard'):
        self.is_training = is_training
        self.scaler = FeatureScaler(method=scaler_method)
        
    def assemble(self, df):
        """Runs the full feature engineering pipeline on a raw dataframe."""
        df_out = df.copy()
        
        # 1. Preprocessing: Timestamp normalization
        if 'date' in df_out.columns:
            df_out = DataPreprocessor.normalize_timestamp(df_out, time_column='date')
            
        # 2. Missing Value Handling & Interpolation
        df_out = DataPreprocessor.interpolate_values(df_out, method='linear')
        df_out = DataPreprocessor.handle_missing_values(df_out, strategy='ffill')
        df_out = DataPreprocessor.handle_missing_values(df_out, strategy='bfill')
        
        # 3. Rolling Window Features
        if 'rainfall' in df_out.columns:
            # e.g., rainfall_3h, rainfall_6h, rainfall_24h
            # Passing 'h' for frequency, so window=3 means 3h
            df_out = add_rolling_sum(df_out, 'rainfall', windows=[3, 6, 24], suffix='h')
            
        # 4. Trend & Rate of Change Features
        if 'soil_moisture' in df_out.columns:
            df_out = calculate_delta(
                df_out, 'soil_moisture', periods=1, target_name='soil_moisture_delta'
            )
            
        if 'humidity' in df_out.columns:
            df_out = add_trend_features(
                df_out, 'humidity', window=6, target_name='humidity_trend'
            )
            
        # 5. Lag Features (Historical context)
        if 'temperature' in df_out.columns:
            df_out = add_lag_features(df_out, 'temperature', lags=[1, 24])

        # Note: 'hotspot_density' and 'elevation_risk' are preserved as-is if present.

        # 6. Scaling
        # Define numerical columns to scale, excluding targets and specific static features
        exclude_cols = ['hazard_label', 'date', 'hotspot_density', 'elevation_risk']
        cols_to_scale = [c for c in df_out.columns if c not in exclude_cols]
        
        if self.is_training:
            df_out = self.scaler.fit_transform(df_out, cols_to_scale)
        else:
            df_out = self.scaler.transform(df_out, cols_to_scale)
            
        # Drop any NaNs caused by lagging/shifting windows at the beginning of the series
        df_out = df_out.dropna()
        
        return df_out
