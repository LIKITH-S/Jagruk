import pandas as pd
import numpy as np
from .simulation_utils import (
    generate_temperature,
    generate_rainfall,
    generate_humidity,
    generate_soil_moisture,
    generate_hotspot_density,
    add_noise,
    calculate_rolling_average,
    calculate_lag,
    determine_hazard_label
)

class SyntheticDataGenerator:
    """
    Generates time-series environmental data for predictive modeling.
    """
    def __init__(self, start_date='2020-01-01', days=1000, seed=42):
        self.start_date = pd.to_datetime(start_date)
        self.days = days
        self.seed = seed
        np.random.seed(self.seed)
        
    def generate_dataset(self):
        """Orchestrates the dataset generation pipeline."""
        df = self._generate_base_features()
        df = self._add_derived_features(df)
        df = self._add_labels(df)
        return df
        
    def _generate_base_features(self):
        dates = pd.date_range(start=self.start_date, periods=self.days, freq='D')
        days_array = np.arange(self.days)
        
        temperature = generate_temperature(days_array)
        rainfall = generate_rainfall(days_array)
        humidity = generate_humidity(rainfall, temperature)
        soil_moisture = generate_soil_moisture(rainfall)
        hotspot_density = generate_hotspot_density(temperature, humidity, soil_moisture)
        
        # Simulating elevation risk as a static region attribute with minor noise
        elevation_risk = np.clip(np.full(self.days, 45.0) + add_noise(self.days, scale=2.0), 0, 100)
        
        df = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'rainfall': rainfall,
            'humidity': humidity,
            'soil_moisture': soil_moisture,
            'hotspot_density': hotspot_density,
            'elevation_risk': elevation_risk
        })
        return df

    def _add_derived_features(self, df):
        # Calculate rolling averages
        for col in ['temperature', 'rainfall', 'humidity']:
            df[f'{col}_roll_7d'] = calculate_rolling_average(df[col], window=7)
            
        # Calculate lag values
        for col in ['soil_moisture', 'hotspot_density', 'elevation_risk']:
            df[f'{col}_lag_1d'] = calculate_lag(df[col], lag=1)
            df[f'{col}_lag_3d'] = calculate_lag(df[col], lag=3)
            
        return df

    def _add_labels(self, df):
        df['hazard_label'] = determine_hazard_label(df)
        return df
