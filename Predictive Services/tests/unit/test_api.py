import pytest
from httpx import AsyncClient

@pytest.mark.anyio
async def test_api_health_endpoint(client: AsyncClient):
    """Verifies the health endpoint returns 200 and standard healthy response."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "predictive-service"
    assert "uptime_seconds" in body
    assert "components" in body

@pytest.mark.anyio
async def test_api_metrics_endpoint(client: AsyncClient):
    """Verifies the Prometheus metrics endpoint is reachable and returns scrape data."""
    resp = await client.get("/metrics/")
    assert resp.status_code == 200
    assert "predictive_service" in resp.text

@pytest.mark.anyio
async def test_api_prediction_validation_fail(client: AsyncClient):
    """Verifies that invalid latitude/longitude or fields trigger validation (422)."""
    invalid_payload = {
        "latitude": 150.0,  # Range must be [-90, 90]
        "longitude": 77.5,
        "timestamp": "2026-05-29T12:00:00Z",
        "hazard_type": "flood"
    }
    resp = await client.post("/api/v1/predict", json=invalid_payload)
    assert resp.status_code == 422
