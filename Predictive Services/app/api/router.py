"""
api/router.py — Aggregates all versioned sub-routers.
Import include_routers() in main.py.
"""
from __future__ import annotations

from fastapi import FastAPI

from app.api.v1 import health as health_v1
from app.api.v1 import predict as predict_v1
from app.config import settings


def include_routers(app: FastAPI) -> None:
    """Mount all versioned routers onto the application instance."""
    prefix = settings.API_PREFIX  # /api/v1

    app.include_router(health_v1.router, prefix=prefix)
    app.include_router(predict_v1.router, prefix=prefix)
