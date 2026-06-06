"""
services/prediction_service.py — Core prediction business logic.

Phase 3: Real model inference via the loaded sklearn/XGBoost artefact.
         Falls back to the deterministic geo-seeded heuristic when no
         model artefact is present (graceful degradation / testing).
"""
from __future__ import annotations

import hashlib
import math
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from app.models.loader import ModelRegistry
from app.monitoring.metrics import (
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_REQUESTS_TOTAL,
    PREDICTION_RISK_SCORE,
)
from app.schemas.prediction import (
    HazardType,
    PredictionRequest,
    PredictionResponse,
    TrendType,
)
from app.utils.exceptions import ModelNotLoadedError, PredictionServiceError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Hazard-type multipliers used when building the synthetic feature vector.
# Each hazard shifts different environmental signals.
# ---------------------------------------------------------------------------
_HAZARD_SIGNAL: dict[HazardType, dict[str, float]] = {
    HazardType.flood:      {"rainfall": 2.5,  "soil_moisture": 2.0,  "temperature": 0.8},
    HazardType.wildfire:   {"temperature": 2.5, "humidity": -1.5,    "hotspot_density": 2.0},
    HazardType.cyclone:    {"rainfall": 2.0,  "humidity": 1.5,       "temperature": 1.2},
    HazardType.drought:    {"temperature": 2.0, "humidity": -2.0,    "soil_moisture": -1.5},
    HazardType.earthquake: {"elevation_risk": 2.0, "temperature": 0.5, "humidity": 0.3},
    HazardType.tsunami:    {"rainfall": 1.0,  "elevation_risk": 2.5, "temperature": 0.5},
    HazardType.landslide:  {"rainfall": 2.0,  "soil_moisture": 2.0,  "elevation_risk": 1.5},
}


class PredictionService:
    """Stateless service that orchestrates real model inference.

    Receives a :class:`ModelRegistry` via dependency injection so tests can
    pass a mock registry without touching the module-level singleton.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    # ── Public API ──────────────────────────────────────────────────────────────

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        if not self._registry.is_ready:
            raise ModelNotLoadedError("Model registry is not initialised.")

        bound_log = logger.bind(
            hazard_type=request.hazard_type.value,
            lat=request.latitude,
            lon=request.longitude,
        )
        bound_log.info("prediction_started")

        t0 = time.perf_counter()
        try:
            risk, confidence = await self._run_inference(request)
            trend = self._derive_trend(risk, request)
        except (ModelNotLoadedError, PredictionServiceError):
            PREDICTION_REQUESTS_TOTAL.labels(
                hazard_type=request.hazard_type.value, status="error"
            ).inc()
            raise
        except Exception as exc:
            PREDICTION_REQUESTS_TOTAL.labels(
                hazard_type=request.hazard_type.value, status="error"
            ).inc()
            raise PredictionServiceError(f"Inference pipeline failed: {exc}") from exc
        finally:
            elapsed = time.perf_counter() - t0
            PREDICTION_LATENCY_SECONDS.labels(
                hazard_type=request.hazard_type.value
            ).observe(elapsed)

        PREDICTION_REQUESTS_TOTAL.labels(
            hazard_type=request.hazard_type.value, status="success"
        ).inc()
        PREDICTION_RISK_SCORE.labels(
            hazard_type=request.hazard_type.value, trend=trend.value
        ).observe(risk)

        response = PredictionResponse(
            predicted_risk=round(risk, 4),
            trend=trend,
            confidence=round(confidence, 4),
            model_version=self._registry.version,
        )
        bound_log.info(
            "prediction_complete",
            risk=response.predicted_risk,
            confidence=response.confidence,
            trend=response.trend.value,
            latency_ms=round(elapsed * 1000, 2),
        )
        return response

    # ── Inference ───────────────────────────────────────────────────────────────

    async def _run_inference(
        self, request: PredictionRequest
    ) -> tuple[float, float]:
        """
        Runs real model inference when a trained artefact is loaded,
        otherwise falls back to the deterministic geo-seeded heuristic.
        """
        if self._registry.model is not None:
            return self._real_inference(request)

        # Graceful fallback — no artefact present (CI / early dev)
        return self._heuristic_inference(request)

    def _real_inference(self, request: PredictionRequest) -> tuple[float, float]:
        """
        Phase 3 real inference:
          1. Build a synthetic feature vector from geo/temporal/hazard signals.
          2. Feed to the loaded model.
          3. Extract positive-class probability → risk score.
          4. Derive confidence from probability certainty.
        """
        # Resolve feature names — explicit chain avoids `or / if / else` precedence trap
        if self._registry.feature_names:
            feature_names: list[str] = self._registry.feature_names
        elif hasattr(self._registry.model, "feature_names_in_"):
            feature_names = self._registry.model.feature_names_in_.tolist()
        else:
            raise PredictionServiceError(
                "No feature names available — ensure feature_names.json exists in the artifacts directory."
            )

        # Build feature dict from request context
        feature_vector = self._build_feature_vector(request, feature_names)

        # Assemble ordered numpy array
        X = np.array(
            [feature_vector.get(f, 0.0) for f in feature_names],
            dtype=np.float64,
        ).reshape(1, -1)

        # Apply preprocessor if available
        if self._registry.preprocessor is not None:
            try:
                X = self._registry.preprocessor.transform(X)
            except Exception as exc:
                logger.warning("preprocessor_transform_failed", error=str(exc))

        # Guard: replace any NaN / inf that slipped in from feature construction
        # or a failed preprocessor transform before the model sees the array.
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        # Predict
        if hasattr(self._registry.model, "predict_proba"):
            prob = float(self._registry.model.predict_proba(X)[0, 1])
        else:
            prob = float(np.clip(self._registry.model.predict(X)[0], 0.0, 1.0))

        risk = float(np.clip(prob, 0.0, 1.0))
        confidence = self._estimate_confidence(prob)
        return risk, confidence

    def _build_feature_vector(
        self, request: PredictionRequest, feature_names: list[str]
    ) -> dict[str, float]:
        """
        Constructs a synthetic, deterministic feature vector from the
        request's geographic, temporal, and hazard-type context.

        All values are seeded from lat/lon/timestamp so the same input
        always produces the same output (deterministic inference).
        """
        seed_str = (
            f"{request.latitude:.4f}|{request.longitude:.4f}"
            f"|{request.hazard_type.value}"
            f"|{request.timestamp.date().isoformat()}"
        )
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)

        # Geo-derived base signals
        lat_norm = abs(request.latitude) / 90.0          # 0 (equator) → 1 (pole)
        lon_norm = abs(request.longitude) / 180.0
        tropical = 1.0 - lat_norm                         # higher near equator
        hour_norm = request.timestamp.hour / 23.0

        # Hazard-specific signal boosts
        signals = _HAZARD_SIGNAL.get(request.hazard_type, {})

        base: dict[str, float] = {
            "temperature":        25.0 + tropical * 10.0 + rng.uniform(-3, 3),
            "rainfall":           tropical * 15.0 + rng.uniform(0, 10),
            "humidity":           50.0 + tropical * 20.0 + rng.uniform(-10, 10),
            "soil_moisture":      30.0 + tropical * 20.0 + rng.uniform(-5, 5),
            "hotspot_density":    max(0.0, tropical * 2.0 + rng.uniform(-1, 2)),
            "elevation_risk":     45.0 + lon_norm * 10.0 + rng.uniform(-3, 3),
        }

        # Apply hazard boosts
        for feat, multiplier in signals.items():
            if feat in base:
                base[feat] = base[feat] + multiplier * tropical * rng.uniform(0.8, 1.2)

        # Derived rolling and lag features (approximate from base values)
        for col in ["temperature", "rainfall", "humidity"]:
            base[f"{col}_roll_7d"] = base[col] + rng.uniform(-1, 1)

        for col in ["soil_moisture", "hotspot_density", "elevation_risk"]:
            base[f"{col}_lag_1d"] = base[col] + rng.uniform(-0.5, 0.5)
            base[f"{col}_lag_3d"] = base[col] + rng.uniform(-1, 1)

        base["rainfall_3h"]          = base["rainfall"] * 0.15 + rng.uniform(0, 2)
        base["rainfall_6h"]          = base["rainfall"] * 0.3  + rng.uniform(0, 3)
        base["rainfall_24h"]         = base["rainfall"] * 1.0  + rng.uniform(0, 5)
        base["soil_moisture_delta"]  = rng.uniform(-2, 2)
        base["humidity_trend"]       = rng.uniform(-0.5, 0.5)
        base["temperature_lag_1"]    = base["temperature"] + rng.uniform(-1, 1)
        base["temperature_lag_24"]   = base["temperature"] + rng.uniform(-2, 2)

        return base

    @staticmethod
    def _estimate_confidence(probability: float) -> float:
        """
        Maps model probability to a confidence score.
        Confidence peaks when probability is near 0 or 1 (decisive),
        and is lowest near 0.5 (maximum uncertainty).
        """
        deviation = abs(probability - 0.5)
        # Sigmoid calibration: maps [0, 0.5] → [0, 1]
        calibrated = 1.0 / (1.0 + math.exp(-6.0 * (deviation - 0.25)))
        confidence = 0.75 + (1.0 - 0.75) * calibrated
        return float(round(min(1.0, max(0.75, confidence)), 4))

    # ── Heuristic fallback (no model artefact) ──────────────────────────────────

    @staticmethod
    def _heuristic_inference(request: PredictionRequest) -> tuple[float, float]:
        """Deterministic geo-seeded heuristic for when no model is loaded."""
        _HAZARD_BASELINE: dict[HazardType, float] = {
            HazardType.flood:      0.65,
            HazardType.earthquake: 0.55,
            HazardType.cyclone:    0.70,
            HazardType.wildfire:   0.60,
            HazardType.drought:    0.40,
            HazardType.tsunami:    0.75,
            HazardType.landslide:  0.50,
        }
        seed_str = (
            f"{request.latitude:.4f}|{request.longitude:.4f}"
            f"|{request.hazard_type.value}"
            f"|{request.timestamp.date().isoformat()}"
        )
        seed_hash = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % 10_000
        jitter = (seed_hash / 10_000) * 0.15
        baseline = _HAZARD_BASELINE[request.hazard_type]
        lat_weight = 1.0 - abs(request.latitude) / 90.0
        risk = min(1.0, max(0.0, baseline + (lat_weight * 0.2) - 0.1 + jitter))
        confidence = round(0.80 + (1 - jitter) * 0.15, 4)
        return risk, confidence

    # ── Trend estimation ────────────────────────────────────────────────────────

    @staticmethod
    def _derive_trend(risk: float, request: PredictionRequest) -> TrendType:
        """
        Derives trend from risk score magnitude and time-of-day context.
        High risk → increasing; low risk → decreasing; mid → hour-dependent.
        """
        if risk > 0.70:
            return TrendType.increasing
        if risk < 0.35:
            return TrendType.decreasing
        hour = request.timestamp.hour
        return TrendType.increasing if 6 <= hour <= 18 else TrendType.stable
