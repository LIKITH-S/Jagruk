"""
api/v1/predict.py — /predict endpoint.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import ORJSONResponse

from app.schemas.prediction import PredictionRequest, PredictionResponse, ErrorResponse
from app.services.dependencies import get_prediction_service
from app.services.prediction_service import PredictionService
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Predict disaster severity risk",
    description=(
        "Accepts environmental observation signals and returns a risk score, "
        "confidence estimate, trend direction, and model version for the "
        "specified hazard type and location."
    ),
    response_class=ORJSONResponse,
)
async def predict(
    request: Request,
    payload: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    # Bind a correlation ID so all log lines for this request share an ID
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    import structlog
    structlog.contextvars.bind_contextvars(request_id=request_id)

    return await service.predict(payload)
