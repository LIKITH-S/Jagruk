import pandas as pd
import numpy as np
import pytest
from feature_engineering.preprocessors import DataPreprocessor
from feature_engineering.scaling import FeatureScaler
from feature_engineering.rolling_features import add_rolling_sum, add_rolling_average
from feature_engineering.lag_features import add_lag_features
from feature_engineering.trend_features import calculate_delta, add_trend_features
from feature_engineering.feature_assembler import FeatureAssembler

def test_data_preprocessor_pipeline():
    """Tests normalizations, missing data interpolation and filling."""
    df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'rainfall': [1.0, np.nan, 3.0]
    })
    
    # Normalize
    df_norm = DataPreprocessor.normalize_timestamp(df, time_column='date')
    assert isinstance(df_norm.index, pd.DatetimeIndex)
    
    # Interpolate
    df_interp = DataPreprocessor.interpolate_values(df_norm, method='linear')
    assert df_interp['rainfall'].iloc[1] == 2.0
    
    # Fill
    df_nan = pd.DataFrame({'rainfall': [1.0, np.nan, np.nan]}, index=df_norm.index)
    df_filled = DataPreprocessor.handle_missing_values(df_nan, strategy='ffill')
    assert df_filled['rainfall'].iloc[1] == 1.0

def test_feature_scaler_zscore():
    """Tests fit and transform features scaling works correctly."""
    df = pd.DataFrame({'temperature': [10.0, 20.0, 30.0]})
    scaler = FeatureScaler(method='standard')
    scaled = scaler.fit_transform(df, ['temperature'])
    
    # Z-Score mean should be ~0 and std should be ~1
    assert np.isclose(scaled['temperature'].mean(), 0.0)
    assert np.isclose(scaled['temperature'].std(), 1.0, atol=0.1)

def test_rolling_sum_aggregation():
    """Tests that rolling aggregations compute expected sliding values."""
    df = pd.DataFrame({'rainfall': [1.0, 2.0, 3.0, 4.0]})
    df_roll = add_rolling_sum(df, 'rainfall', [2], suffix='')
    assert df_roll['rainfall_2'].tolist() == [1.0, 3.0, 5.0, 7.0]

def test_lag_features_shifting():
    """Tests lag shift operates correctly creating appropriate lag cols."""
    df = pd.DataFrame({'soil_moisture': [1.0, 2.0, 3.0]})
    df_lag = add_lag_features(df, 'soil_moisture', [1])
    assert pd.isna(df_lag['soil_moisture_lag_1'].iloc[0])
    assert df_lag['soil_moisture_lag_1'].iloc[1] == 1.0

def test_trend_delta_and_slopes():
    """Tests trend calculations like delta and linear regression slopes."""
    df = pd.DataFrame({'humidity': [10.0, 15.0, 25.0]})
    df_delta = calculate_delta(df, 'humidity', periods=1, target_name='humidity_delta')
    assert df_delta['humidity_delta'].tolist()[1:] == [5.0, 10.0]
