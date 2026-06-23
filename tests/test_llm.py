"""Gemini path: the request carries the configured thinking level, empty output fails loudly,
and a live-LLM failure degrades to the offline template rather than crashing the brief."""

import json
import logging

import duckdb
import httpx
import pytest
import respx

import mmi.settings as settings_mod
from mmi.ai import llm, narrative

_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"


@respx.mock
def test_gemini_request_carries_configured_thinking_level(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings_mod.settings, "gemini_api_key", "k")
    monkeypatch.setattr(settings_mod.settings, "gemini_thinking_level", "high")
    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        )
    )

    assert llm.complete("prompt", system="sys") == "hi"  # parses the answer text
    body = json.loads(route.calls.last.request.content)
    assert (
        body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"
    )  # reflects settings


@respx.mock
def test_gemini_raises_when_thinking_consumes_all_tokens(monkeypatch):
    # finishReason=MAX_TOKENS with no answer parts: must fail loudly (not IndexError on parts[0]).
    monkeypatch.setattr(settings_mod.settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings_mod.settings, "gemini_api_key", "k")
    respx.post(_URL).mock(
        return_value=httpx.Response(200, json={"candidates": [{"finishReason": "MAX_TOKENS"}]})
    )
    with pytest.raises(RuntimeError, match="no text"):
        llm._gemini("p", None, 100)


def _boom(*_args, **_kwargs):
    raise RuntimeError("simulated LLM failure")


def test_generate_brief_falls_back_to_template_on_llm_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(narrative.llm, "available", lambda: True)
    monkeypatch.setattr(narrative.llm, "complete", _boom)
    monkeypatch.setattr(settings_mod.settings, "duckdb_path", tmp_path / "x.duckdb")
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    try:
        text = narrative.generate_brief(con)
        engine = con.execute("select engine from marts.market_brief").fetchone()[0]
    finally:
        con.close()
    assert text.strip()  # the template floor still produced a brief
    assert "llm-failed" in engine  # provenance records the degraded path


def _raise_with_key(*_args, **_kwargs):
    # Mimics an httpx error string: the Gemini key rides in the request URL query param.
    raise RuntimeError(
        "Client error '400' for url 'https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3.5-flash:generateContent?key=SUPERSECRET123'"
    )


def test_generate_brief_redacts_provider_key_in_failure_log(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(narrative.llm, "available", lambda: True)
    monkeypatch.setattr(narrative.llm, "complete", _raise_with_key)
    monkeypatch.setattr(settings_mod.settings, "duckdb_path", tmp_path / "x.duckdb")
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    try:
        with caplog.at_level(logging.WARNING):
            narrative.generate_brief(con)
    finally:
        con.close()
    assert "SUPERSECRET123" not in caplog.text  # the key never reaches the logs
    assert "key=***" in caplog.text  # redact() ran on the exception string


@respx.mock
def test_complete_does_not_retry_a_deterministic_empty_output(monkeypatch):
    # The empty-parts RuntimeError is not an httpx error, so the retry filter must NOT retry it —
    # one call, fail fast (vs. burning 3x backoff + free quota on a deterministic failure).
    monkeypatch.setattr(settings_mod.settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings_mod.settings, "gemini_api_key", "k")
    route = respx.post(_URL).mock(
        return_value=httpx.Response(200, json={"candidates": [{"finishReason": "MAX_TOKENS"}]})
    )
    with pytest.raises(RuntimeError, match="no text"):
        llm.complete("p")
    assert route.call_count == 1  # not retried
