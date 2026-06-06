"""
metrics.py - Prometheus monitoring wrapper for standard application instrumentation.
"""
from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Latency histograms
PREDICTION_LATENCY = Histogram(
    name="predictive_service_prediction_latency_seconds",
    documentation="Latency of model training or inference runs in seconds.",
    labelnames=["model_type"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Counter metrics
PREDICTION_COUNT = Counter(
    name="predictive_service_prediction_requests_total",
    documentation="Total number of prediction tasks or calls made.",
    labelnames=["model_type", "status"]
)

# Operational status gauge
MODEL_STATUS = Gauge(
    name="predictive_service_model_loaded",
    documentation="System model health indicator (1 for active, 0 for idle).",
    labelnames=["model_type"]
)

def get_latest_metrics() -> tuple[bytes, str]:
    """Generates the latest Prometheus registry metrics in text format."""
    return generate_latest(), CONTENT_TYPE_LATEST
