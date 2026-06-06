"""
monitoring/metrics.py — Prometheus metrics registry for the predictive service.

All metrics are defined here as module-level singletons.
Import and use them directly from anywhere in the codebase.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

# ── Prediction metrics ────────────────────────────────────────────────────────

PREDICTION_REQUESTS_TOTAL = Counter(
    name="predictive_service_prediction_requests_total",
    documentation="Total number of prediction requests received.",
    labelnames=["hazard_type", "status"],   # status: success | error
)

PREDICTION_LATENCY_SECONDS = Histogram(
    name="predictive_service_prediction_latency_seconds",
    documentation="End-to-end prediction request latency in seconds.",
    labelnames=["hazard_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_RISK_SCORE = Histogram(
    name="predictive_service_prediction_risk_score",
    documentation="Distribution of predicted risk scores [0, 1].",
    labelnames=["hazard_type", "trend"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ── HTTP metrics ───────────────────────────────────────────────────────────────

HTTP_REQUESTS_TOTAL = Counter(
    name="predictive_service_http_requests_total",
    documentation="Total HTTP requests received.",
    labelnames=["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    name="predictive_service_http_request_duration_seconds",
    documentation="HTTP request processing duration in seconds.",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# ── Model / system metrics ────────────────────────────────────────────────────

MODEL_LOADED = Gauge(
    name="predictive_service_model_loaded",
    documentation="1 if the ML model is loaded and ready, 0 otherwise.",
)

MODEL_LOAD_DURATION_SECONDS = Gauge(
    name="predictive_service_model_load_duration_seconds",
    documentation="Time taken to load the model artefact at startup.",
)

CACHE_HIT_TOTAL = Counter(
    name="predictive_service_cache_hit_total",
    documentation="Total Redis cache hits for prediction results.",
    labelnames=["hazard_type"],
)

CACHE_MISS_TOTAL = Counter(
    name="predictive_service_cache_miss_total",
    documentation="Total Redis cache misses for prediction results.",
    labelnames=["hazard_type"],
)
