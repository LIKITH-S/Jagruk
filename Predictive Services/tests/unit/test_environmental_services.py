"""
test_environmental_services.py — Unit tests for environmental data fetchers.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from app.services.rainfall_service import RainfallService
from app.services.weather_service import WeatherService
from app.services.soil_moisture_service import SoilMoistureService
from app.services.fire_hotspot_service import FireHotspotService
from app.services.elevation_service import ElevationService


@pytest.fixture
def test_coords() -> tuple[float, float, datetime]:
    return 12.9, 77.5, datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.anyio
async def test_rainfall_service_mock(test_coords):
    lat, lon, ts = test_coords
    service = RainfallService()
    # Force mock mode
    service.mock_mode = True
    
    data = await service.fetch_rainfall(lat, lon, ts)
    assert "precipitation_24h" in data
    assert "cumulative_7d" in data
    assert data["mocked"] is True


@pytest.mark.anyio
async def test_weather_service_mock(test_coords):
    lat, lon, ts = test_coords
    service = WeatherService()
    service.mock_mode = True
    
    data = await service.fetch_weather(lat, lon, ts)
    assert "temperature_c" in data
    assert "humidity_percent" in data
    assert data["mocked"] is True


@pytest.mark.anyio
async def test_soil_moisture_service_mock(test_coords):
    lat, lon, ts = test_coords
    service = SoilMoistureService()
    service.mock_mode = True
    
    data = await service.fetch_soil_moisture(lat, lon, ts)
    assert "volumetric_water_content_surface" in data
    assert "volumetric_water_content_rootzone" in data
    assert data["mocked"] is True


@pytest.mark.anyio
async def test_fire_hotspot_service_mock(test_coords):
    lat, lon, ts = test_coords
    service = FireHotspotService()
    service.mock_mode = True
    
    data = await service.fetch_hotspots(lat, lon, ts)
    assert "active_hotspots_count" in data
    assert "max_fire_radiative_power_mw" in data
    assert data["mocked"] is True


@pytest.mark.anyio
async def test_elevation_service_mock(test_coords):
    lat, lon, _ = test_coords
    service = ElevationService()
    service.mock_mode = True
    
    data = await service.fetch_elevation(lat, lon)
    assert "elevation_meters" in data
    assert "slope_degrees" in data
    assert data["mocked"] is True
