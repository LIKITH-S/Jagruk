"""
confidence_estimator.py
Estimates prediction confidence from model probability output.
"""
from __future__ import annotations

import numpy as np


class ConfidenceEstimator:
    """
    Converts raw model probability into a calibrated confidence score.

    Strategy:
      - Confidence peaks when probability is near 0.0 or 1.0 (model is decisive).
      - Confidence is lowest when probability ≈ 0.5 (maximum uncertainty).
      - Applies a sigmoid-based calibration curve.
    """

    def __init__(self, base_confidence: float = 0.75, scale: float = 6.0):
        """
        Args:
            base_confidence: Minimum confidence floor (default 0.75).
            scale: Steepness of the sigmoid calibration (default 6.0).
        """
        self.base_confidence = base_confidence
        self.scale = scale

    def estimate(self, probability: float) -> float:
        """
        Computes confidence score from a raw class probability.

        Args:
            probability: Float in [0, 1] — predicted probability of positive class.

        Returns:
            Confidence score in [base_confidence, 1.0].
        """
        # Distance from the maximum-uncertainty point (0.5)
        deviation = abs(probability - 0.5)

        # Sigmoid calibration: maps [0, 0.5] deviation → [0, 1]
        calibrated = 1.0 / (1.0 + np.exp(-self.scale * (deviation - 0.25)))

        # Scale to [base_confidence, 1.0]
        confidence = self.base_confidence + (1.0 - self.base_confidence) * calibrated
        return float(round(min(1.0, max(self.base_confidence, confidence)), 4))
