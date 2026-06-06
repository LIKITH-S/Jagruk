"""
services/soil_moisture_service.py — Soil moisture telemetry service.
Retrieves surface and root-zone soil saturation levels.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import structlog

from app.services.http_client import AsyncHTTPClient
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SoilMoistureService:
    """
    Fetches real-time and historical agricultural and geological soil sensors.
    """

    def __init__(self, client: AsyncHTTPClient | None = None) -> None:
        self.api_url = os.getenv("SOIL_MOISTURE_API_URL", "https://api.disaster-telemetry.io/v1/soil")
        self.api_key = os.getenv("SOIL_MOISTURE_API_KEY", "")
        self.mock_mode = os.getenv("MOCK_ENVIRONMENTAL_SERVICES", "true").lower() == "true"
        
        self.client = client or AsyncHTTPClient(
            base_url=self.api_url,
            timeout=float(os.getenv("SOIL_MOISTURE_API_TIMEOUT", "5.0")),
            max_retries=int(os.getenv("SOIL_MOISTURE_API_RETRIES", "3")),
        )

    async def fetch_soil_moisture(self, lat: float, lon: float, timestamp: datetime) -> dict[str, Any]:
        """
        Retrieves volumetric water content (VWC) telemetry.
        """
        log = logger.bind(lat=lat, lon=lon, timestamp=timestamp.isoformat())
        
        if self.mock_mode:
            log.debug("soil_moisture_fetch_mock_mode_active")
            return self._generate_mock_response(lat, lon, timestamp)

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            params = {
                "lat": lat,
                "lon": lon,
                "time": timestamp.isoformat(),
            }
            response = await self.client.request("GET", "/moisture", params=params, headers=headers)
            data = response.json()
            log.info("soil_moisture_fetch_success", vwc_surface=data.get("volumetric_water_content_surface"))
            return data
        except Exception as exc:
            log.warning("soil_moisture_fetch_failed_using_fallback", error=str(exc))
            return self._generate_mock_response(lat, lon, timestamp, is_fallback=True)

    def _generate_mock_response(
        self, lat: float, lon: float, timestamp: datetime, is_fallback: bool = False
    ) -> dict[str, Any]:
        """
        Generates realistic soil volumetric water content (VWC) data.
        """
        # Coordinate base determination
        coordinate_hash = int(abs(lat * 100) + abs(lon * 50)) % 100
        
        # Volumetric Water Content generally ranges from 0.05 (very dry) to 0.45 (saturated)
        vwc_surface = 0.05 + (coordinate_hash / 250.0)  # max ~0.45
        vwc_rootzone = 0.08 + (coordinate_hash / 300.0)
        
        return {
            "volumetric_water_content_surface": round(vwc_surface, 3),
            "volumetric_water_content_rootzone": round(vwc_rootzone, 3),
            "soil_temperature_10cm_c": round(20.0 - (lat / 5.0), 1),
            "saturation_percentage": round((vwc_surface / 0.5) * 100, 1),
            "mocked": True,
            "fallback": is_fallback,
            "retrieved_at": datetime.utcnow().isoformat() + "Z",
        }
