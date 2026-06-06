"""
services/fire_hotspot_service.py — Fire hotspot telemetry service.
Detects active thermal anomalies or wildfires in target bounding coordinates.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import structlog

from app.services.http_client import AsyncHTTPClient
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FireHotspotService:
    """
    Retrieves satellite-based active wildfire / thermal radiation anomalies.
    """

    def __init__(self, client: AsyncHTTPClient | None = None) -> None:
        self.api_url = os.getenv("FIRE_API_URL", "https://api.disaster-telemetry.io/v1/fire")
        self.api_key = os.getenv("FIRE_API_KEY", "")
        self.mock_mode = os.getenv("MOCK_ENVIRONMENTAL_SERVICES", "true").lower() == "true"
        
        self.client = client or AsyncHTTPClient(
            base_url=self.api_url,
            timeout=float(os.getenv("FIRE_API_TIMEOUT", "5.0")),
            max_retries=int(os.getenv("FIRE_API_RETRIES", "3")),
        )

    async def fetch_hotspots(self, lat: float, lon: float, timestamp: datetime, radius_km: float = 10.0) -> dict[str, Any]:
        """
        Retrieves active thermal anomalies / fire radiative power index inside radius.
        """
        log = logger.bind(lat=lat, lon=lon, radius_km=radius_km)
        
        if self.mock_mode:
            log.debug("fire_hotspot_fetch_mock_mode_active")
            return self._generate_mock_response(lat, lon, radius_km)

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            params = {
                "lat": lat,
                "lon": lon,
                "radius": radius_km,
                "time": timestamp.isoformat(),
            }
            response = await self.client.request("GET", "/hotspots", params=params, headers=headers)
            data = response.json()
            log.info("fire_hotspot_fetch_success", active_hotspots=data.get("active_hotspots_count"))
            return data
        except Exception as exc:
            log.warning("fire_hotspot_fetch_failed_using_fallback", error=str(exc))
            return self._generate_mock_response(lat, lon, radius_km, is_fallback=True)

    def _generate_mock_response(
        self, lat: float, lon: float, radius_km: float, is_fallback: bool = False
    ) -> dict[str, Any]:
        """
        Generates realistic satellite thermal anomaly measurements.
        """
        # Coordinate-based pseudo-random generator
        coordinate_hash = int(abs(lat * 10) + abs(lon * 20)) % 100
        
        # Most areas will have 0 active fires unless they cross specific coordinate markers
        has_fires = coordinate_hash > 88
        active_hotspots = int((coordinate_hash % 5) + 1) if has_fires else 0
        
        # Radiative Power in Megawatts
        max_frp = round((coordinate_hash * 2.5), 1) if active_hotspots > 0 else 0.0

        return {
            "active_hotspots_count": active_hotspots,
            "max_fire_radiative_power_mw": max_frp,
            "mean_confidence_score": round(0.50 + (coordinate_hash / 200.0), 2) if active_hotspots > 0 else 0.0,
            "radius_searched_km": radius_km,
            "mocked": True,
            "fallback": is_fallback,
            "retrieved_at": datetime.utcnow().isoformat() + "Z",
        }
