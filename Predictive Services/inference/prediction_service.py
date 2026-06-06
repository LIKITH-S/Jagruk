"""
prediction_service.py
Stateless prediction service that wraps a loaded model for single-sample inference.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from inference.confidence_estimator import ConfidenceEstimator
from inference.trend_analyzer import Trend, TrendAnalyzer
from inference.explainability_service import ExplainabilityService

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Wraps a fitted model and orchestrates:
      - Feature vector validation
      - Probability prediction
      - Confidence estimation
      - Trend analysis
      - Explainability
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_version: str = "v1",
        confidence_estimator: Optional[ConfidenceEstimator] = None,
        trend_analyzer: Optional[TrendAnalyzer] = None,
        explainability_service: Optional[ExplainabilityService] = None,
    ):
        self.model           = model
        self.feature_names   = feature_names
        self.model_version   = model_version
        self._confidence     = confidence_estimator or ConfidenceEstimator()
        self._trend          = trend_analyzer or TrendAnalyzer()
        self._explainability = explainability_service or ExplainabilityService(model, feature_names)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(
        self,
        feature_vector: Dict[str, float],
        history: Optional[List[float]] = None,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs a full prediction for a single sample.

        Args:
            feature_vector: Dict mapping feature name → float value.
            history:        Optional list of prior risk probabilities (most recent last).
            explain:        Whether to include feature-level explanations.

        Returns:
            Dict matching the standard response schema:
            {
                "predicted_risk": float,
                "trend": str,
                "confidence": float,
                "model_version": str,
                "explain": {"top_features": [...]}
            }
        """
        X = self._build_feature_array(feature_vector)

        # --- Risk probability ---
        probability = self._get_probability(X)
        predicted_risk = round(float(probability), 4)

        # --- Confidence ---
        confidence = self._confidence.estimate(probability)

        # --- Trend ---
        trend: Trend = self._trend.analyze(probability, history=history)

        # --- Explain ---
        top_features: List[Dict] = []
        if explain:
            top_features = self._explainability.explain(X)

        response = {
            "predicted_risk": predicted_risk,
            "trend":          trend.value,
            "confidence":     confidence,
            "model_version":  self.model_version,
            "explain":        {"top_features": top_features},
        }
        logger.info(
            "Prediction complete | risk=%.4f | confidence=%.4f | trend=%s",
            predicted_risk, confidence, trend.value,
        )
        return response

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_feature_array(self, feature_vector: Dict[str, float]) -> np.ndarray:
        """Aligns input dict to model's expected feature order."""
        missing = [f for f in self.feature_names if f not in feature_vector]
        if missing:
            logger.warning("Missing features (filled with 0.0): %s", missing)

        row = np.array(
            [feature_vector.get(f, 0.0) for f in self.feature_names],
            dtype=np.float64,
        )
        return row.reshape(1, -1)

    def _get_probability(self, X: np.ndarray) -> float:
        """Returns positive-class probability from the model."""
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X)[0, 1]
        else:
            # Hard-prediction fallback (e.g., SVMs without probability calibration)
            prob = float(self.model.predict(X)[0])
        return float(np.clip(prob, 0.0, 1.0))
