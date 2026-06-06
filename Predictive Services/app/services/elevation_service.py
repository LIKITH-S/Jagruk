"""
services/elevation_service.py — Elevation and topographic telemetry service.
Retrieves altitude profile, terrain slope, and rugosity indices for specified coordinates.
"""
from __future__ import annotations

import os
from typing import Any

import structlog

from app.services.http_client import AsyncHTTPClient
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ElevationService:
    """
    Fetches elevation and topography metrics for geological flood risk and landslide mapping.
    """

    def __init__(self, client: AsyncHTTPClient | None = None) -> None:
        self.api_url = os.getenv("ELEVATION_API_URL", "https://api.disaster-telemetry.io/v1/elevation")
        self.api_key = os.getenv("ELEVATION_API_KEY", "")
        self.mock_mode = os.getenv("MOCK_ENVIRONMENTAL_SERVICES", "true").lower() == "true"
        
        self.client = client or AsyncHTTPClient(
            base_url=self.api_url,
            timeout=float(os.getenv("ELEVATION_API_TIMEOUT", "5.0")),
            max_retries=int(os.getenv("ELEVATION_API_RETRIES", "3")),
        )

    async def fetch_elevation(self, lat: float, lon: float) -> dict[str, Any]:
        """
        Retrieves absolute elevation and slope parameters.
        """
        log = logger.bind(lat=lat, lon=lon)
        
        if self.mock_mode:
            log.debug("elevation_fetch_mock_mode_active")
            return self._generate_mock_response(lat, lon)

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            params = {
                "lat": lat,
                "lon": lon,
            }
            response = await self.client.request("GET", "/profile", params=params, headers=headers)
            data = response.json()
            log.info("elevation_fetch_success", elevation_m=data.get("elevation_meters"))
            return data
        except Exception as exc:
            log.warning("elevation_fetch_failed_using_fallback", error=str(exc))
            return self._generate_mock_response(lat, lon, is_fallback=True)

    def _generate_mock_response(self, lat: float, lon: float, is_fallback: bool = False) -> dict[str, Any]:
        """
        Generates realistic topological and geological profile data.
        """
        # Coordinate-based calculation to make altitude deterministic
        coordinate_hash = int(abs(lat * 73) + abs(lon * 29)) % 1000
        
        # Ranges from sea level to mountains
        elevation_meters = float(12 + coordinate_hash * 2.5)  # 12 to 2512 meters
        slope_degrees = round((coordinate_hash % 45) * 0.8, 1)  # 0 to 36 degrees slope

        return {
            "elevation_meters": elevation_meters,
            "slope_degrees": slope_degrees,
            "rugosity_index": round(1.0 + (slope_degrees / 90.0), 3),
            "aspect_direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][coordinate_hash % 8],
            "mocked": True,
            "fallback": is_fallback,
        }
