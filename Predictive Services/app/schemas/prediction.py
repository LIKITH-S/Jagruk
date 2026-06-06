"""
schemas/prediction.py — Request and response contracts for the /predict endpoint.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# ── Enumerations ──────────────────────────────────────────────────────────────

class HazardType(str, Enum):
    flood      = "flood"
    earthquake = "earthquake"
    cyclone    = "cyclone"
    wildfire   = "wildfire"
    drought    = "drought"
    tsunami    = "tsunami"
    landslide  = "landslide"


class TrendType(str, Enum):
    increasing = "increasing"
    decreasing = "decreasing"
    stable     = "stable"


# ── Request ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "latitude":    12.9,
                "longitude":   77.5,
                "timestamp":   "2026-05-29T12:00:00Z",
                "hazard_type": "flood",
            }
        }
    )

    latitude:    float      = Field(..., ge=-90.0,   le=90.0,   description="WGS-84 latitude")
    longitude:   float      = Field(..., ge=-180.0,  le=180.0,  description="WGS-84 longitude")
    timestamp:   datetime   = Field(..., description="Observation timestamp (ISO-8601, UTC preferred)")
    hazard_type: HazardType = Field(..., description="Type of disaster hazard to evaluate")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc_aware(cls, v: datetime | str) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v


# ── Response ──────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id":  "a3f1c2d4-...",
                "predicted_risk": 0.82,
                "trend":          "increasing",
                "confidence":     0.91,
                "model_version":  "v1",
                "processed_at":   "2026-05-29T12:00:01.234Z",
            }
        }
    )

    prediction_id:  str       = Field(default_factory=lambda: str(uuid.uuid4()))
    predicted_risk: float     = Field(..., ge=0.0, le=1.0, description="Risk score [0, 1]")
    trend:          TrendType = Field(..., description="Detected risk trajectory")
    confidence:     float     = Field(..., ge=0.0, le=1.0, description="Model confidence [0, 1]")
    model_version:  str       = Field(..., description="Artefact version tag")
    processed_at:   datetime  = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when prediction was generated",
    )


# ── Error response (used by exception handlers) ───────────────────────────────

class ErrorDetail(BaseModel):
    code:    str
    message: str
    field:   str | None = None


class ErrorResponse(BaseModel):
    status:  int
    errors:  list[ErrorDetail]
    request_id: str | None = None
