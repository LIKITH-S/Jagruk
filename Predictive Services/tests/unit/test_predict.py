"""test_predict.py — /predict endpoint tests."""
from __future__ import annotations

import pytest


VALID_PAYLOAD = {
    "latitude":    12.9,
    "longitude":   77.5,
    "timestamp":   "2026-05-29T12:00:00Z",
    "hazard_type": "flood",
}


@pytest.mark.anyio
async def test_predict_returns_200(client):
    resp = await client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_predict_response_schema(client):
    resp = await client.post("/api/v1/predict", json=VALID_PAYLOAD)
    body = resp.json()
    assert 0.0 <= body["predicted_risk"] <= 1.0
    assert body["trend"] in ("increasing", "decreasing", "stable")
    assert 0.0 <= body["confidence"] <= 1.0
    assert "model_version" in body
    assert "prediction_id" in body
    assert "processed_at" in body


@pytest.mark.anyio
async def test_predict_deterministic(client):
    """Same input must yield the same risk score (placeholder is hash-based)."""
    r1 = await client.post("/api/v1/predict", json=VALID_PAYLOAD)
    r2 = await client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert r1.json()["predicted_risk"] == r2.json()["predicted_risk"]


@pytest.mark.anyio
@pytest.mark.parametrize("hazard", [
    "flood", "earthquake", "cyclone", "wildfire", "drought", "tsunami", "landslide"
])
async def test_predict_all_hazard_types(client, hazard):
    payload = {**VALID_PAYLOAD, "hazard_type": hazard}
    resp = await client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_predict_invalid_latitude(client):
    resp = await client.post(
        "/api/v1/predict", json={**VALID_PAYLOAD, "latitude": 999}
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_predict_invalid_hazard_type(client):
    resp = await client.post(
        "/api/v1/predict", json={**VALID_PAYLOAD, "hazard_type": "volcano"}
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_predict_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "timestamp"}
    resp = await client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_predict_request_id_propagated(client):
    resp = await client.post(
        "/api/v1/predict",
        json=VALID_PAYLOAD,
        headers={"X-Request-ID": "test-correlation-id-123"},
    )
    assert resp.headers.get("X-Request-ID") == "test-correlation-id-123"
