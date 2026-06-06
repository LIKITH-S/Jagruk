"""
monitoring/middleware.py — ASGI logging + Prometheus HTTP middleware.

Injected into the FastAPI middleware stack in main.py.
"""
from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.monitoring.metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL

logger = structlog.get_logger(__name__)

# Paths excluded from access logging (noisy probes)
_SILENT_PATHS = frozenset({"/api/v1/health/live", "/api/v1/health/ready", "/metrics"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured access log + Prometheus HTTP metrics per request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        t0 = time.perf_counter()
        response: Response | None = None

        try:
            response = await call_next(request)
            return response
        except Exception:
            raise
        finally:
            elapsed = time.perf_counter() - t0
            status_code = response.status_code if response is not None else 500
            endpoint = request.url.path

            # Update Prometheus metrics
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(elapsed)

            # Structured access log (skip silent paths)
            if endpoint not in _SILENT_PATHS:
                logger.info(
                    "http_request",
                    method=request.method,
                    path=endpoint,
                    status_code=status_code,
                    duration_ms=round(elapsed * 1000, 2),
                    client=request.client.host if request.client else "unknown",
                    request_id=request_id,
                )

            # Propagate the X-Request-ID to the caller
            if response is not None:
                response.headers["X-Request-ID"] = request_id
