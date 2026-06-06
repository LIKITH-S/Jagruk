"""
main.py — FastAPI application factory.

Wires together:
  - Lifespan events  (model load / unload)
  - Middleware stack  (logging, CORS, GZip)
  - API routers      (/api/v1/...)
  - Prometheus ASGI  (/metrics)
  - Exception handlers
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

from app.api.router import include_routers
from app.config import settings
from app.models.loader import registry
from app.monitoring.middleware import RequestLoggingMiddleware
from app.utils.exceptions import register_exception_handlers
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle manager (replaces deprecated on_event)."""
    configure_logging()
    logger.info(
        "service_starting",
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENV,
    )

    # Load ML artefacts — safe to call even when .joblib files are absent (Phase 2)
    await registry.load()

    logger.info("service_ready", name=settings.APP_NAME)
    yield  # ← application runs here

    logger.info("service_shutting_down", name=settings.APP_NAME)
    await registry.unload()
    logger.info("service_stopped")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Predictive Service",
        version=settings.APP_VERSION,
        description=(
            "Disaster severity prediction service. "
            "Ingests environmental and satellite signals; "
            "delivers risk scores to the Fusion Engine."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    # ── Middleware (outermost → innermost) ────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.ENV == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(RequestLoggingMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── API routers ───────────────────────────────────────────────────────────
    include_routers(app)

    # ── Prometheus /metrics ASGI sub-app ──────────────────────────────────────
    if settings.METRICS_ENABLED:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    return app


# Module-level instance used by uvicorn / gunicorn
app: FastAPI = create_app()
