"""The provider key must never reach the logs on the SUCCESS path.

Gemini (and FRED) pass the API key as a URL query param, so httpx's own logger emits
``HTTP Request: POST .../generateContent?key=<KEY> "HTTP/1.1 200 OK"`` at INFO on every
successful request. ``redact()`` only scrubs EXCEPTION strings, so the happy path is not
covered by it — the leak is stopped by clamping the httpx logger (see utils/logging.py).
This test reproduces a real successful ``_gemini`` call and asserts the key is absent from
INFO-level logs.
"""

from __future__ import annotations

import logging

import httpx

from mmi.ai import llm

_SECRET = "AIzaSy-TEST-SECRET-KEY-do-not-log-0123456789"


def test_successful_gemini_call_never_logs_the_api_key(monkeypatch, caplog):
    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "gemini_api_key", _SECRET)

    captured_url: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # The key really does travel in the outgoing URL — this is the leak surface httpx logs.
        captured_url["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
            request=request,
        )

    # Use a real httpx.Client wired to a MockTransport so httpx's *own* per-request INFO
    # logging fires exactly as on the live path — only the network call is faked.
    real_client = httpx.Client
    monkeypatch.setattr(
        llm.httpx,
        "Client",
        lambda *a, **k: real_client(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )

    with caplog.at_level(logging.INFO):
        assert llm._gemini("hello", None, 16) == "ok"

    # Guard against a false pass: confirm the secret was genuinely in the request URL...
    assert _SECRET in captured_url["url"]
    # ...yet never reached the captured logs.
    assert _SECRET not in caplog.text
    # The clamp is what suppresses it — assert it directly so this test fails loudly if the
    # httpx logger is ever un-clamped (even should httpx change its log message format).
    assert logging.getLogger("httpx").level == logging.WARNING
