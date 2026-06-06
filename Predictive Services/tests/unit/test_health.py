"""test_health.py — Health endpoint tests."""
from __future__ import annotations

import pytest


@pytest.mark.anyio
async def test_health_returns_200(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_health_body_structure(client):
    resp = await client.get("/api/v1/health")
    body = resp.json()
    assert body["service"] == "predictive-service"
    assert "uptime_seconds" in body
    assert "components" in body
    assert "model_registry" in body["components"]


@pytest.mark.anyio
async def test_liveness_probe(client):
    resp = await client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


@pytest.mark.anyio
async def test_readiness_probe_when_ready(client):
    resp = await client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
