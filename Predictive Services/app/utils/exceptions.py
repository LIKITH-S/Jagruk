"""
utils/exceptions.py — Custom exception types and FastAPI exception handlers.
Register all handlers in main.py via register_exception_handlers(app).
"""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse

from app.schemas.prediction import ErrorDetail, ErrorResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before the model is ready."""


class PredictionServiceError(RuntimeError):
    """Raised when the prediction pipeline fails unexpectedly."""


class FusionEngineError(RuntimeError):
    """Raised on upstream Fusion Engine communication failure."""


# ── Helper ────────────────────────────────────────────────────────────────────

def _request_id(request: Request) -> str | None:
    return request.headers.get("X-Request-ID")


def _json(status_code: int, errors: list[ErrorDetail], request: Request) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            status=status_code,
            errors=errors,
            request_id=_request_id(request),
        ).model_dump(mode="json"),
    )


# ── Handlers ──────────────────────────────────────────────────────────────────

async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> ORJSONResponse:
    errors = [
        ErrorDetail(
            code="VALIDATION_ERROR",
            message=e["msg"],
            field=".".join(str(loc) for loc in e["loc"]),
        )
        for e in exc.errors()
    ]
    logger.warning("request_validation_failed", path=request.url.path, errors=len(errors))
    return _json(status.HTTP_422_UNPROCESSABLE_ENTITY, errors, request)


async def model_not_loaded_handler(
    request: Request, exc: ModelNotLoadedError
) -> ORJSONResponse:
    logger.error("model_not_loaded", path=request.url.path)
    return _json(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        [ErrorDetail(code="MODEL_NOT_LOADED", message=str(exc))],
        request,
    )


async def prediction_service_handler(
    request: Request, exc: PredictionServiceError
) -> ORJSONResponse:
    logger.error("prediction_service_error", path=request.url.path, error=str(exc))
    return _json(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        [ErrorDetail(code="PREDICTION_FAILED", message=str(exc))],
        request,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> ORJSONResponse:
    logger.exception("unhandled_exception", path=request.url.path, error=str(exc))
    return _json(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        [ErrorDetail(code="INTERNAL_ERROR", message="An unexpected error occurred.")],
        request,
    )


# ── Registration helper ───────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
    app.add_exception_handler(PredictionServiceError, prediction_service_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
