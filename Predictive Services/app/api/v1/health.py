"""
api/v1/health.py — Detailed health endpoint.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from fastapi.responses import ORJSONResponse

from app.models.loader import ModelRegistry
from app.schemas.health import ComponentHealth, ComponentStatus, HealthResponse
from app.services.dependencies import get_model_registry
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])

# Module-level startup time so uptime can be computed
_SERVICE_START_TIME: float = time.time()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns liveness + readiness status for all internal components.",
    response_class=ORJSONResponse,
)
async def health(
    registry: ModelRegistry = Depends(get_model_registry),
) -> HealthResponse:
    uptime = time.time() - _SERVICE_START_TIME
    model_status = ComponentStatus.ok if registry.is_ready else ComponentStatus.unhealthy

    overall = (
        ComponentStatus.ok
        if registry.is_ready
        else ComponentStatus.degraded
    )

    return HealthResponse(
        status=overall,
        service="predictive-service",
        version=registry.version,
        environment=__import__("app.config", fromlist=["settings"]).settings.ENV,
        uptime_seconds=round(uptime, 2),
        model_loaded=registry.is_ready,
        components={
            "model_registry": ComponentHealth(
                status=model_status,
                detail=f"version={registry.version}",
            ),
            # Redis and Fusion Engine probes are added in Phase 4
        },
    )


@router.get(
    "/live",
    summary="Liveness probe",
    description="Kubernetes liveness check — returns 200 if the process is alive.",
    response_class=ORJSONResponse,
    status_code=status.HTTP_200_OK,
)
async def liveness() -> dict:
    return {"status": "alive"}


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Kubernetes readiness check — returns 200 only when the model is loaded.",
    response_class=ORJSONResponse,
)
async def readiness(
    registry: ModelRegistry = Depends(get_model_registry),
) -> ORJSONResponse:
    if registry.is_ready:
        return ORJSONResponse(content={"status": "ready"}, status_code=status.HTTP_200_OK)
    return ORJSONResponse(
        content={"status": "not_ready", "reason": "model_loading"},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )
