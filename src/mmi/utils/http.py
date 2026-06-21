"""HTTP helpers with retry/backoff — important when living on free API tiers."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mmi.utils.logging import get_logger

log = get_logger("http")

_RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def get_json(url: str, *, params: dict | None = None, headers: dict | None = None) -> Any:
    """GET a URL and return parsed JSON, retrying on transient errors / 429s."""
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def get_text(url: str, *, params: dict | None = None) -> str:
    """GET a URL and return the response body as text (used for CSV endpoints)."""
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.text
