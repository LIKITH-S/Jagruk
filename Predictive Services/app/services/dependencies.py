"""
services/dependencies.py — FastAPI dependency injection providers.

Import these functions in route handlers with `Depends(...)`.
"""
from __future__ import annotations

from fastapi import Depends

from app.models.loader import registry, ModelRegistry
from app.services.prediction_service import PredictionService


def get_model_registry() -> ModelRegistry:
    """Provide the module-level model registry singleton."""
    return registry


def get_prediction_service(
    reg: ModelRegistry = Depends(get_model_registry),
) -> PredictionService:
    """Construct a PredictionService with the injected registry."""
    return PredictionService(reg)
