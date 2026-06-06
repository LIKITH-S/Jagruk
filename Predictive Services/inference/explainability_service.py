"""
explainability_service.py
Provides feature-level explanations for predictions.

Strategy (in priority order):
  1. SHAP TreeExplainer — if `shap` is installed and model is tree-based.
  2. Built-in feature_importances_ — available on sklearn / XGBoost models.
  3. Fallback: returns empty list with a warning.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """
    Lightweight explainability layer with SHAP + fallback support.
    """

    def __init__(self, model: Any, feature_names: List[str], top_n: int = 10):
        """
        Args:
            model:         Fitted sklearn/XGBoost model.
            feature_names: Ordered list of feature names matching model input.
            top_n:         Number of top features to return.
        """
        self.model         = model
        self.feature_names = feature_names
        self.top_n         = top_n
        self._shap_explainer = None
        self._try_init_shap()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def explain(self, X_row: np.ndarray) -> List[Dict[str, Any]]:
        """
        Explains a single prediction row.

        Args:
            X_row: 1-D or 2-D numpy array (single sample).

        Returns:
            List of dicts: [{"feature": str, "importance": float, "value": float}]
        """
        X = np.atleast_2d(X_row)

        if self._shap_explainer is not None:
            return self._explain_with_shap(X)

        return self._explain_with_feature_importance(X)

    def global_importances(self) -> List[Dict[str, Any]]:
        """Returns global model-level feature importances (non-sample-specific)."""
        if not hasattr(self.model, "feature_importances_"):
            return []
        importances = self.model.feature_importances_
        ranked = sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": f, "importance": round(v, 6)} for f, v in ranked[: self.top_n]]

    # ------------------------------------------------------------------
    # SHAP path
    # ------------------------------------------------------------------

    def _try_init_shap(self) -> None:
        try:
            import shap  # noqa: F401
            self._shap_explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialised.")
        except ImportError:
            logger.info("shap not installed — using feature_importances_ fallback.")
        except Exception as exc:
            logger.warning("SHAP init failed (%s) — using fallback.", exc)

    def _explain_with_shap(self, X: np.ndarray) -> List[Dict[str, Any]]:
        import shap
        shap_values = self._shap_explainer.shap_values(X)

        # For binary classifiers, shap_values is [class0, class1]
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        ranked = sorted(
            zip(self.feature_names, values.tolist(), X[0].tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return [
            {"feature": f, "importance": round(imp, 6), "value": round(val, 4)}
            for f, imp, val in ranked[: self.top_n]
        ]

    def _explain_with_feature_importance(self, X: np.ndarray) -> List[Dict[str, Any]]:
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Model has no feature_importances_ attribute.")
            return []

        importances = self.model.feature_importances_
        ranked = sorted(
            zip(self.feature_names, importances.tolist(), X[0].tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {"feature": f, "importance": round(imp, 6), "value": round(val, 4)}
            for f, imp, val in ranked[: self.top_n]
        ]
