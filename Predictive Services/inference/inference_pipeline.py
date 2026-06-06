"""
inference_pipeline.py
Top-level orchestrator: loads model + meta from disk, wires all inference
components together, exposes a single .run() entrypoint.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

from inference.confidence_estimator import ConfidenceEstimator
from inference.explainability_service import ExplainabilityService
from inference.prediction_service import PredictionService
from inference.trend_analyzer import TrendAnalyzer

logger = logging.getLogger(__name__)

# Default artifact locations (relative to project root)
_DEFAULT_MODEL_PATH = os.path.join("artifacts", "model.pkl")
_DEFAULT_META_PATH  = os.path.join("artifacts", "model_meta.json")


class InferencePipeline:
    """
    Loads a trained model and meta once at startup, then serves predictions
    through a clean .run() interface.

    Usage:
        pipeline = InferencePipeline()
        pipeline.load()
        result   = pipeline.run(feature_vector, history=None, explain=True)
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        meta_path:  str = _DEFAULT_META_PATH,
    ):
        self.model_path = model_path
        self.meta_path  = meta_path
        self._service: Optional[PredictionService] = None
        self._meta:    Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "InferencePipeline":
        """Loads model + meta from disk and wires all sub-components."""
        logger.info("Loading model from %s", self.model_path)
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        logger.info("Loading meta from %s", self.meta_path)
        with open(self.meta_path, "r") as f:
            self._meta = json.load(f)

        feature_names: List[str] = self._meta.get("feature_names", [])
        model_version: str       = self._meta.get("model_type", "v1")

        self._service = PredictionService(
            model=model,
            feature_names=feature_names,
            model_version=model_version,
            confidence_estimator=ConfidenceEstimator(),
            trend_analyzer=TrendAnalyzer(),
            explainability_service=ExplainabilityService(model, feature_names),
        )
        logger.info(
            "InferencePipeline ready | model=%s | features=%d",
            model_version, len(feature_names),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        feature_vector: Dict[str, float],
        history: Optional[List[float]] = None,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs the full prediction pipeline for a single sample.

        Args:
            feature_vector: Dict[feature_name → float].
            history:        Optional list of prior risk probabilities (recency-ascending).
            explain:        Whether to include feature-level explanations.

        Returns:
            Standard prediction response dict.
        """
        if self._service is None:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        return self._service.predict(
            feature_vector=feature_vector,
            history=history,
            explain=explain,
        )

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta

    @property
    def is_ready(self) -> bool:
        return self._service is not None


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def build_pipeline(
    model_path: str = _DEFAULT_MODEL_PATH,
    meta_path:  str = _DEFAULT_META_PATH,
) -> InferencePipeline:
    """Creates and loads a pipeline in one call."""
    return InferencePipeline(model_path=model_path, meta_path=meta_path).load()
