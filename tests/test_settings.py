"""Settings-loading unit tests: the env-overridable knobs parsed by Pydantic Settings.

Covers the two positive-int knobs migrated from call-site ``os.environ`` parsing
(GO_LIVE_PLAN D1/D6): defaults, env alias precedence, and the defensive
warn-and-fall-back contract (a fat-fingered value never crashes the run).
"""

import logging

import pytest

from mmi.settings import Settings


def _clear_knobs(monkeypatch):
    monkeypatch.delenv("MMI_PORTFOLIO_N_BOOT", raising=False)
    monkeypatch.delenv("MMI_SNAPSHOT_MAX_BYTES", raising=False)


def test_knob_defaults(monkeypatch):
    _clear_knobs(monkeypatch)
    s = Settings(_env_file=None)
    assert s.portfolio_n_boot == 2000
    assert s.snapshot_max_bytes == 12_000_000


def test_knobs_read_env_aliases(monkeypatch):
    _clear_knobs(monkeypatch)
    monkeypatch.setenv("MMI_PORTFOLIO_N_BOOT", "42")
    monkeypatch.setenv("MMI_SNAPSHOT_MAX_BYTES", "7")
    s = Settings(_env_file=None)
    assert s.portfolio_n_boot == 42
    assert s.snapshot_max_bytes == 7


@pytest.mark.parametrize("bad_value", ["", "abc", "2000.5", "0", "-5"])
def test_portfolio_n_boot_invalid_falls_back_with_warning(monkeypatch, caplog, bad_value):
    _clear_knobs(monkeypatch)
    monkeypatch.setenv("MMI_PORTFOLIO_N_BOOT", bad_value)
    with caplog.at_level(logging.WARNING, logger="mmi.settings"):
        s = Settings(_env_file=None)
    assert s.portfolio_n_boot == 2000
    assert any("MMI_PORTFOLIO_N_BOOT" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("bad_value", ["", "abc", "2.5", "0", "-5"])
def test_snapshot_max_bytes_invalid_falls_back_with_warning(monkeypatch, caplog, bad_value):
    _clear_knobs(monkeypatch)
    monkeypatch.setenv("MMI_SNAPSHOT_MAX_BYTES", bad_value)
    with caplog.at_level(logging.WARNING, logger="mmi.settings"):
        s = Settings(_env_file=None)
    assert s.snapshot_max_bytes == 12_000_000
    assert any("MMI_SNAPSHOT_MAX_BYTES" in r.getMessage() for r in caplog.records)


def test_knobs_do_not_read_other_env_vars(monkeypatch):
    """Unrelated env vars must not bleed into the knob fields."""
    _clear_knobs(monkeypatch)
    monkeypatch.setenv("MMI_LOG_LEVEL", "DEBUG")
    s = Settings(_env_file=None)
    assert s.portfolio_n_boot == 2000
    assert s.snapshot_max_bytes == 12_000_000
