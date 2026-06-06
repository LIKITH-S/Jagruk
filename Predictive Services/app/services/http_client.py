"""
services/http_client.py — Shared async HTTP utility client.
Handles retries, timeouts, logging, and graceful exception wrapping.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Mapping

import httpx
import structlog

from app.utils.exceptions import FusionEngineError  # Reusable base/ecosystem exception wrapper

logger = structlog.get_logger(__name__)


class AsyncHTTPClient:
    """
    Generic reusable async HTTP client with timeout, retry logic,
    and structured logging.
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        # Create limits to prevent connection exhaustion
        self.limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)

    async def request(
        self,
        method: str,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json_data: Any = None,
    ) -> httpx.Response:
        """
        Executes an HTTP request with retries and exponential backoff.
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        client_headers = dict(headers or {})
        
        # Propagate request id if in context
        try:
            import structlog
            context = structlog.contextvars.get_contextvars()
            if "request_id" in context:
                client_headers["X-Request-ID"] = context["request_id"]
        except Exception:
            pass

        async with httpx.AsyncClient(limits=self.limits, timeout=self.timeout) as client:
            last_exception = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.request(
                        method=method,
                        url=url,
                        params=params,
                        headers=client_headers,
                        json=json_data,
                    )
                    # Raise for 4xx/5xx to trigger retry or error logging
                    response.raise_for_status()
                    return response
                except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                    last_exception = exc
                    
                    # Do not retry on client errors (4xx) except for occasional rate limits (429)
                    if isinstance(exc, httpx.HTTPStatusError) and 400 <= exc.response.status_code < 500:
                        if exc.response.status_code != 429:
                            logger.error(
                                "http_client_client_error",
                                url=url,
                                status_code=exc.response.status_code,
                                attempt=attempt,
                            )
                            raise exc

                    logger.warning(
                        "http_client_request_retry",
                        url=url,
                        attempt=attempt,
                        max_retries=self.max_retries,
                        error=str(exc),
                    )
                    
                    if attempt < self.max_retries:
                        sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                        await asyncio.sleep(sleep_time)
            
            # If all retries failed
            logger.error(
                "http_client_request_failed_all_retries",
                url=url,
                max_retries=self.max_retries,
                error=str(last_exception),
            )
            raise last_exception or httpx.HTTPError("Request failed after retries")
