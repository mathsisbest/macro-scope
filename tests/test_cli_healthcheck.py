"""Tests for mmi healthcheck — no network; probe seam is mocked.

Covers:
- classify_source: skip_reason set -> "skip" AND probe NOT called
- classify_source: probe ok -> "ok"
- classify_source: probe raises -> "fail"
- exit_code: required+fail -> 1; optional+fail -> 0; skip -> 0
- Probe exception with embedded secret -> redacted in ProbeResult.detail
- cmd_healthcheck returns exit_code (monkeypatches EXTRACTORS, no DB)
"""

from __future__ import annotations

import argparse

import pandas as pd

import mmi.cli as cli
import mmi.ingestion as ingestion
import mmi.utils.db as db_mod
from mmi.ingestion.base import Extractor
from mmi.ingestion.healthcheck import (
    ProbeResult,
    classify_source,
    exit_code,
    format_table,
    run_healthcheck,
)

# ---------------------------------------------------------------------------
# Stub extractors
# ---------------------------------------------------------------------------


class _OkRequired(Extractor):
    """Required source whose probe succeeds."""

    source = "ok_required"
    table = "raw.ok_required"
    keys = ["a"]
    required = True
    probe_url = "https://example.com/ok"
    _probe_called = False

    def probe(self) -> None:
        type(self)._probe_called = True  # record that probe was invoked

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1]})


class _OkOptional(Extractor):
    """Optional source whose probe succeeds."""

    source = "ok_optional"
    table = "raw.ok_optional"
    keys = ["a"]
    required = False
    probe_url = "https://example.com/ok-optional"
    _probe_called = False

    def probe(self) -> None:
        type(self)._probe_called = True

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1]})


class _FailRequired(Extractor):
    """Required source whose probe raises."""

    source = "fail_required"
    table = "raw.fail_required"
    keys = ["a"]
    required = True
    probe_url = "https://example.com/fail"

    def probe(self) -> None:
        raise RuntimeError("network error")

    def fetch(self) -> pd.DataFrame:
        raise RuntimeError("never called in healthcheck")


class _FailOptional(Extractor):
    """Optional source whose probe raises."""

    source = "fail_optional"
    table = "raw.fail_optional"
    keys = ["a"]
    required = False
    probe_url = "https://example.com/fail-optional"

    def probe(self) -> None:
        raise RuntimeError("optional down")

    def fetch(self) -> pd.DataFrame:
        raise RuntimeError("never called in healthcheck")


class _SkipRequired(Extractor):
    """Required source that skips (key absent)."""

    source = "skip_required"
    table = "raw.skip_required"
    keys = ["a"]
    required = True
    probe_url = "https://example.com/skip"
    _probe_called = False

    def skip_reason(self) -> str | None:
        return "API_KEY not set"

    def probe(self) -> None:
        type(self)._probe_called = True  # must NOT be reached

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1]})


class _FailWithSecret(Extractor):
    """Source whose probe raises an exception that embeds an API key."""

    source = "fail_secret"
    table = "raw.fail_secret"
    keys = ["a"]
    required = True
    probe_url = "https://api.stlouisfed.org/fred/series/observations"

    def probe(self) -> None:
        raise RuntimeError(
            "400 Bad Request for url "
            "'https://api.stlouisfed.org/fred/series/observations"
            "?series_id=DGS10&api_key=SECRETXYZ123&file_type=json'"
        )

    def fetch(self) -> pd.DataFrame:
        raise RuntimeError("never called in healthcheck")


# ---------------------------------------------------------------------------
# classify_source tests
# ---------------------------------------------------------------------------


def test_classify_skip_reason_set_returns_skip_and_probe_not_called():
    """skip_reason() non-None -> status="skip"; probe() must not be called."""
    _SkipRequired._probe_called = False
    result = classify_source(_SkipRequired(loader=None))  # type: ignore[arg-type]

    assert result.status == "skip"
    assert result.source == "skip_required"
    assert result.required is True
    assert "API_KEY not set" in result.detail
    assert _SkipRequired._probe_called is False, "probe must not be called when skip_reason is set"


def test_classify_probe_ok_returns_ok():
    """probe() succeeds -> status="ok" with empty detail."""
    _OkRequired._probe_called = False
    result = classify_source(_OkRequired(loader=None))  # type: ignore[arg-type]

    assert result.status == "ok"
    assert result.source == "ok_required"
    assert result.required is True
    assert result.detail == ""
    assert _OkRequired._probe_called is True


def test_classify_probe_raises_returns_fail():
    """probe() raises -> status="fail" with detail set."""
    result = classify_source(_FailRequired(loader=None))  # type: ignore[arg-type]

    assert result.status == "fail"
    assert result.source == "fail_required"
    assert result.required is True
    assert "network error" in result.detail


def test_classify_probe_exception_secret_is_redacted():
    """Secret in probe exception message is redacted in ProbeResult.detail."""
    result = classify_source(_FailWithSecret(loader=None))  # type: ignore[arg-type]

    assert result.status == "fail"
    assert "SECRETXYZ123" not in result.detail, "raw secret must not appear in detail"
    assert "api_key=***" in result.detail, "masked form must be present in detail"


# ---------------------------------------------------------------------------
# exit_code tests
# ---------------------------------------------------------------------------


def test_exit_code_required_fail_returns_1():
    results = [ProbeResult(source="x", status="fail", required=True, detail="err")]
    assert exit_code(results) == 1


def test_exit_code_optional_fail_returns_0():
    results = [ProbeResult(source="x", status="fail", required=False, detail="err")]
    assert exit_code(results) == 0


def test_exit_code_skip_required_returns_0():
    results = [ProbeResult(source="x", status="skip", required=True, detail="no key")]
    assert exit_code(results) == 0


def test_exit_code_all_ok_returns_0():
    results = [
        ProbeResult(source="a", status="ok", required=True),
        ProbeResult(source="b", status="ok", required=False),
    ]
    assert exit_code(results) == 0


def test_exit_code_mixed_optional_fail_and_ok_returns_0():
    results = [
        ProbeResult(source="a", status="ok", required=True),
        ProbeResult(source="b", status="fail", required=False, detail="err"),
    ]
    assert exit_code(results) == 0


def test_exit_code_any_required_fail_returns_1():
    results = [
        ProbeResult(source="a", status="ok", required=True),
        ProbeResult(source="b", status="fail", required=True, detail="err"),
    ]
    assert exit_code(results) == 1


# ---------------------------------------------------------------------------
# run_healthcheck tests
# ---------------------------------------------------------------------------


def test_run_healthcheck_preserves_order():
    """Results preserve the input order of extractor classes."""
    results = run_healthcheck([_OkRequired, _FailOptional, _SkipRequired])
    assert [r.source for r in results] == ["ok_required", "fail_optional", "skip_required"]


# ---------------------------------------------------------------------------
# format_table tests
# ---------------------------------------------------------------------------


def test_format_table_contains_all_sources():
    results = [
        ProbeResult(source="fred", status="ok", required=True),
        ProbeResult(source="yahoo", status="skip", required=True, detail="no key"),
        ProbeResult(source="coingecko", status="fail", required=False, detail="err"),
    ]
    table = format_table(results)
    assert "fred" in table
    assert "yahoo" in table
    assert "coingecko" in table
    assert "ok" in table
    assert "skip" in table
    assert "fail" in table


def test_format_table_empty():
    assert "No sources" in format_table([])


# ---------------------------------------------------------------------------
# cmd_healthcheck integration (no DB)
# ---------------------------------------------------------------------------


def test_cmd_healthcheck_returns_0_when_all_ok(monkeypatch):
    """cmd_healthcheck returns 0 when all probes pass."""
    monkeypatch.setattr(ingestion, "EXTRACTORS", [_OkRequired, _OkOptional])
    result = cli.cmd_healthcheck(argparse.Namespace())
    assert result == 0


def test_cmd_healthcheck_returns_1_when_required_fails(monkeypatch):
    """cmd_healthcheck returns 1 when a required source fails."""
    monkeypatch.setattr(ingestion, "EXTRACTORS", [_FailRequired])
    result = cli.cmd_healthcheck(argparse.Namespace())
    assert result == 1


def test_cmd_healthcheck_returns_0_when_only_optional_fails(monkeypatch):
    """cmd_healthcheck returns 0 when only optional sources fail."""
    monkeypatch.setattr(ingestion, "EXTRACTORS", [_OkRequired, _FailOptional])
    result = cli.cmd_healthcheck(argparse.Namespace())
    assert result == 0


def test_cmd_healthcheck_returns_0_when_required_skips(monkeypatch):
    """cmd_healthcheck returns 0 when a required source skips (key absent)."""
    monkeypatch.setattr(ingestion, "EXTRACTORS", [_SkipRequired])
    result = cli.cmd_healthcheck(argparse.Namespace())
    assert result == 0


def test_cmd_healthcheck_no_db_connection(monkeypatch):
    """cmd_healthcheck must not open a DB connection (connect is not called)."""
    connection_opened = []

    def _no_connect(*args, **kwargs):
        connection_opened.append(True)
        raise AssertionError("healthcheck must not open a DB connection")

    monkeypatch.setattr(db_mod, "connect", _no_connect)
    monkeypatch.setattr(ingestion, "EXTRACTORS", [_OkRequired])
    # Should not raise
    cli.cmd_healthcheck(argparse.Namespace())
    assert not connection_opened, "connect() was called — healthcheck must be DB-free"
