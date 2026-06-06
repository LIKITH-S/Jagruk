"""
trend_analyzer.py
Determines risk trend direction from a sequence of probabilities or
from a single probability and its prior context.
"""
from __future__ import annotations

from enum import Enum
from typing import List


class Trend(str, Enum):
    increasing = "increasing"
    decreasing = "decreasing"
    stable     = "stable"


class TrendAnalyzer:
    """
    Computes risk trend from a window of historical probability scores.

    If only a single current probability is available, uses the prior value
    stored internally since the last call.
    """

    def __init__(self, threshold: float = 0.03, window: int = 3):
        """
        Args:
            threshold: Minimum absolute delta to classify as increasing/decreasing.
            window:    Number of recent scores to use for slope estimation.
        """
        self.threshold = threshold
        self.window    = window
        self._history: List[float] = []

    def analyze(self, current_probability: float, history: List[float] | None = None) -> Trend:
        """
        Returns the trend direction based on recent probability scores.

        Args:
            current_probability: Latest model probability output.
            history: Optional external list of prior probabilities (most recent last).
                     If omitted, uses internal rolling history.

        Returns:
            Trend enum value.
        """
        scores = (history[-self.window:] if history else self._history[-self.window:]) + [current_probability]

        # Update internal history
        self._history.append(current_probability)
        if len(self._history) > 20:
            self._history.pop(0)

        if len(scores) < 2:
            return Trend.stable

        # Weighted slope: more weight to recent deltas
        deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        weights = list(range(1, len(deltas) + 1))
        weighted_delta = sum(d * w for d, w in zip(deltas, weights)) / sum(weights)

        if weighted_delta > self.threshold:
            return Trend.increasing
        if weighted_delta < -self.threshold:
            return Trend.decreasing
        return Trend.stable
