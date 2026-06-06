import numpy as np
import pandas as pd

def add_seasonality(days, period=365, amplitude=1.0, phase_shift=0):
    """Generates a sinusoidal seasonal pattern."""
    return amplitude * np.sin(2 * np.pi * (days + phase_shift) / period)

def add_noise(size, scale=0.1):
    """Generates random noise."""
    return np.random.normal(scale=scale, size=size)

def generate_temperature(days_array):
    """Simulates daily temperature with seasonality and noise."""
    base = 25
    seasonality = add_seasonality(days_array, period=365, amplitude=12, phase_shift=-90)
    noise = add_noise(len(days_array), scale=2.5)
    return base + seasonality + noise

def generate_rainfall(days_array):
    """Simulates rainfall with dry/wet seasons and random spikes."""
    seasonality = add_seasonality(days_array, period=365, amplitude=8, phase_shift=90)
    base_rain = np.maximum(0, seasonality + add_noise(len(days_array), scale=4))
    
    # Add random extreme storms
    storms = np.random.exponential(scale=3, size=len(days_array))
    storms[storms < 8] = 0
    return base_rain + storms

def generate_humidity(rainfall, temperature):
    """Humidity correlates positively with rainfall and negatively with high temp."""
    base_humidity = 60
    rain_effect = rainfall * 1.5
    temp_effect = (25 - temperature) * 0.8
    humidity = base_humidity + rain_effect + temp_effect + add_noise(len(rainfall), scale=5)
    return np.clip(humidity, 10, 100)

def generate_soil_moisture(rainfall):
    """Soil moisture increases with rainfall and decays over time."""
    moisture = np.zeros(len(rainfall))
    current_moisture = 30.0
    for i in range(len(rainfall)):
        current_moisture = current_moisture * 0.85 + rainfall[i] * 2.0
        moisture[i] = current_moisture
    return np.clip(moisture + add_noise(len(rainfall), scale=2), 5, 100)

def generate_hotspot_density(temperature, humidity, soil_moisture):
    """Simulates hotspots (e.g., thermal anomalies/fires) under dry, hot conditions."""
    risk_factor = (temperature * 1.5) - (humidity * 0.5) - (soil_moisture * 0.5)
    risk_factor = np.maximum(0, risk_factor - 15)
    density = risk_factor * 0.2 + add_noise(len(temperature), scale=0.5)
    return np.clip(density, 0, None)

def calculate_rolling_average(series, window=7):
    """Calculates rolling average for a pandas Series."""
    return series.rolling(window=window, min_periods=1).mean()

def calculate_lag(series, lag=1):
    """Calculates lag value for a pandas Series."""
    return series.shift(lag).bfill()

def determine_hazard_label(df):
    """
    Determines hazard labels based on extreme environmental conditions.
    0: Normal
    1: Hazard Event
    """
    # Flood risk conditions
    flood_risk = (df['rainfall_roll_7d'] > 20) & (df['soil_moisture'] > 85)
    
    # Fire risk conditions
    fire_risk = (df['temperature_roll_7d'] > 32) & (df['humidity_roll_7d'] < 40) & (df['hotspot_density'] > 3)
    
    # Generic extreme temperature anomaly
    extreme_temp = df['temperature'] > df['temperature'].quantile(0.98)
    
    hazard = (flood_risk | fire_risk | extreme_temp).astype(int)
    return hazard
