"""
schemas/health.py — Health check response schema.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ComponentStatus(str, Enum):
    ok        = "ok"
    degraded  = "degraded"
    unhealthy = "unhealthy"


class ComponentHealth(BaseModel):
    status:     ComponentStatus
    latency_ms: float | None = Field(None, description="Round-trip latency in ms")
    detail:     str   | None = None


class HealthResponse(BaseModel):
    status:          ComponentStatus
    service:         str
    version:         str
    environment:     str
    uptime_seconds:  float
    model_loaded:    bool
    components:      dict[str, ComponentHealth]
    checked_at:      datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
