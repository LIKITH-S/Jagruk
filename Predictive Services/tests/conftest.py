"""
conftest.py — Shared pytest fixtures (Phase 2 update).
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models.loader import registry


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture()
async def client() -> AsyncClient:
    """Async HTTP test client wired to the FastAPI ASGI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def ensure_registry_ready(monkeypatch):
    """Force the model registry into a ready state for all tests
    so we never depend on real .joblib files being present."""
    monkeypatch.setattr(registry, "is_ready", True)
    monkeypatch.setattr(registry, "version", "v1-test")
    monkeypatch.setattr(registry, "model", None)   # keeps placeholder inference path
