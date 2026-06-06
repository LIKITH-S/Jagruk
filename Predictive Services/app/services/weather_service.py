"""
services/weather_service.py — Weather data fetcher service.
Fetches ambient temperature, humidity, wind speed, and atmospheric conditions.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import structlog

from app.services.http_client import AsyncHTTPClient
from app.utils.logging import get_logger

logger = get_logger(__name__)


class WeatherService:
    """
    Fetches real-time and forecasted meteorological telemetry.
    """

    def __init__(self, client: AsyncHTTPClient | None = None) -> None:
        self.api_url = os.getenv("WEATHER_API_URL", "https://api.disaster-telemetry.io/v1/weather")
        self.api_key = os.getenv("WEATHER_API_KEY", "")
        self.mock_mode = os.getenv("MOCK_ENVIRONMENTAL_SERVICES", "true").lower() == "true"
        
        self.client = client or AsyncHTTPClient(
            base_url=self.api_url,
            timeout=float(os.getenv("WEATHER_API_TIMEOUT", "5.0")),
            max_retries=int(os.getenv("WEATHER_API_RETRIES", "3")),
        )

    async def fetch_weather(self, lat: float, lon: float, timestamp: datetime) -> dict[str, Any]:
        """
        Retrieves weather telemetry metrics for a given location and timestamp.
        """
        log = logger.bind(lat=lat, lon=lon, timestamp=timestamp.isoformat())
        
        if self.mock_mode:
            log.debug("weather_fetch_mock_mode_active")
            return self._generate_mock_response(lat, lon, timestamp)

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            params = {
                "lat": lat,
                "lon": lon,
                "time": timestamp.isoformat(),
            }
            response = await self.client.request("GET", "/current", params=params, headers=headers)
            data = response.json()
            log.info("weather_fetch_success", temp_c=data.get("temperature_c"))
            return data
        except Exception as exc:
            log.warning("weather_fetch_failed_using_fallback", error=str(exc))
            return self._generate_mock_response(lat, lon, timestamp, is_fallback=True)

    def _generate_mock_response(
        self, lat: float, lon: float, timestamp: datetime, is_fallback: bool = False
    ) -> dict[str, Any]:
        """
        Generates realistic meteorological data based on geographical zones.
        """
        # Base temperature: warmer near the equator
        dist_from_equator = abs(lat)
        base_temp = 32.0 - (dist_from_equator * 0.4)  # Warm equator, colder poles
        
        # Diurnal fluctuation based on hour
        hour = timestamp.hour
        diurnal_shift = 4.0 * ((hour - 6) / 12.0) if 6 <= hour <= 18 else -2.0
        temp_c = round(max(-10.0, base_temp + diurnal_shift), 1)

        # Humidity is inversely related to diurnal temperature swings
        humidity = int(min(100, max(15, 60 - diurnal_shift * 3 + (abs(lon) % 20))))
        
        # Wind speed based on longitude coordinates
        wind_speed_kmh = round(5.0 + (abs(lon) % 25), 1)

        return {
            "temperature_c": temp_c,
            "humidity_percent": humidity,
            "wind_speed_kmh": wind_speed_kmh,
            "wind_direction_deg": int((abs(lat) * 10) % 360),
            "pressure_hpa": 1013 - int(lat / 10),
            "mocked": True,
            "fallback": is_fallback,
            "retrieved_at": datetime.utcnow().isoformat() + "Z",
        }
