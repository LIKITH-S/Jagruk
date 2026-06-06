import numpy as np
import pytest
from inference.confidence_estimator import ConfidenceEstimator
from inference.trend_analyzer import TrendAnalyzer, Trend
from inference.explainability_service import ExplainabilityService
from inference.prediction_service import PredictionService

class DummyModel:
    """Mock ML model for testing inference pipeline components."""
    def __init__(self):
        self.feature_importances_ = np.array([0.7, 0.3])
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])
    def predict(self, X):
        return np.array([1])

def test_confidence_estimation_logic():
    """Ensures certainty outputs scale with probability extremes."""
    estimator = ConfidenceEstimator(base_confidence=0.7)
    
    # 0.5 probability (highest uncertainty) -> lower confidence
    conf_uncertain = estimator.estimate(0.5)
    # 0.99 probability (highly decisive) -> high confidence
    conf_certain = estimator.estimate(0.99)
    
    assert conf_uncertain < conf_certain
    assert conf_uncertain >= 0.7
    assert conf_certain <= 1.0

def test_trend_analysis_deltas():
    """Verifies trend categorizations base correctly on direction trends."""
    analyzer = TrendAnalyzer(threshold=0.05)
    
    assert analyzer.analyze(0.6, history=[0.5]) == Trend.increasing
    assert analyzer.analyze(0.4, history=[0.5]) == Trend.decreasing
    assert analyzer.analyze(0.51, history=[0.5]) == Trend.stable

def test_explainability_fallback():
    """Checks feature contribution explanations operate smoothly."""
    model = DummyModel()
    explainer = ExplainabilityService(model, ['rainfall', 'temperature'])
    explanations = explainer.explain(np.array([[1.5, 2.5]]))
    
    assert len(explanations) == 2
    assert explanations[0]['feature'] == 'rainfall'
    assert explanations[0]['value'] == 1.5

def test_prediction_orchestration_loop():
    """Verifies that PredictionService coordinates outputs into the response format."""
    model = DummyModel()
    service = PredictionService(model, ['rainfall', 'temperature'], model_version="v1-rf")
    
    payload = {'rainfall': 1.5, 'temperature': 2.5}
    response = service.predict(payload, explain=True)
    
    assert response["predicted_risk"] == 0.8
    assert response["trend"] in [t.value for t in Trend]
    assert 0.0 <= response["confidence"] <= 1.0
    assert response["model_version"] == "v1-rf"
    assert "top_features" in response["explain"]
