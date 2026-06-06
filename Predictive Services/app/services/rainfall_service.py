"""
services/rainfall_service.py — Rainfall data fetcher service.
Fetches precipitation and cumulative rainfall measurements for the given coordinates.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import structlog

from app.services.http_client import AsyncHTTPClient
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RainfallService:
    """
    Fetches real-time and historical rainfall/precipitation data.
    Integrates with environmental telemetry endpoints.
    """

    def __init__(self, client: AsyncHTTPClient | None = None) -> None:
        # Config-driven endpoint configuration via env with safe defaults
        self.api_url = os.getenv("RAINFALL_API_URL", "https://api.disaster-telemetry.io/v1/rainfall")
        self.api_key = os.getenv("RAINFALL_API_KEY", "")
        self.mock_mode = os.getenv("MOCK_ENVIRONMENTAL_SERVICES", "true").lower() == "true"
        
        self.client = client or AsyncHTTPClient(
            base_url=self.api_url,
            timeout=float(os.getenv("RAINFALL_API_TIMEOUT", "5.0")),
            max_retries=int(os.getenv("RAINFALL_API_RETRIES", "3")),
        )

    async def fetch_rainfall(self, lat: float, lon: float, timestamp: datetime) -> dict[str, Any]:
        """
        Retrieves precipitation metrics for the specified location and time.
        Gracefully falls back to high-fidelity mocks on failure or if mock mode is active.
        """
        log = logger.bind(lat=lat, lon=lon, timestamp=timestamp.isoformat())
        
        if self.mock_mode:
            log.debug("rainfall_fetch_mock_mode_active")
            return self._generate_mock_response(lat, lon, timestamp)

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            params = {
                "lat": lat,
                "lon": lon,
                "time": timestamp.isoformat(),
            }
            response = await self.client.request("GET", "/precipitation", params=params, headers=headers)
            data = response.json()
            log.info("rainfall_fetch_success", precipitation_mm=data.get("precipitation_24h"))
            return data
        except Exception as exc:
            log.warning("rainfall_fetch_failed_using_fallback", error=str(exc))
            return self._generate_mock_response(lat, lon, timestamp, is_fallback=True)

    def _generate_mock_response(
        self, lat: float, lon: float, timestamp: datetime, is_fallback: bool = False
    ) -> dict[str, Any]:
        """
        Generates realistic rainfall data based on lat/lon coordinates.
        Tropical zones yield higher default values; incorporates minor date variation.
        """
        # Simple coordinate-based deterministic heuristic
        base_precip = abs(lat) % 5.0  # 0 to 5 mm
        if abs(lat) < 23.5:  # Tropical zone
            base_precip += 8.0

        # Modulate by month to simulate seasonal variations
        seasonal_multiplier = 1.0 + 0.5 * (1 if 5 <= timestamp.month <= 9 else -0.5)
        precip_24h = round(max(0.0, base_precip * seasonal_multiplier), 2)
        
        return {
            "precipitation_24h": precip_24h,
            "cumulative_7d": round(precip_24h * 4.2, 2),
            "intensity": "moderate" if precip_24h > 10 else "light" if precip_24h > 0 else "none",
            "unit": "mm",
            "mocked": True,
            "fallback": is_fallback,
            "retrieved_at": datetime.utcnow().isoformat() + "Z",
        }
