"""`mmi ml-gate` CLI command — unit tests for C6.

These tests monkeypatch ``skill_verdict`` so they exercise the CLI wiring (DB read,
argument parsing, exit-code contract, printed output) without training any real ML model
and without requiring a real DB with rv_har metrics.

Three scenarios are covered:
  1. skill_verdict returns cleared=True  -> exit 0
  2. skill_verdict returns cleared=False -> exit non-zero + reasons printed
  3. --warn-only + not cleared           -> exit 0 (warn-only contract)
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import duckdb

import mmi.cli as cli

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path, *, with_metrics: bool = True) -> duckdb.DuckDBPyConnection:
    """Return a real DuckDB connection with an optionally-populated model_metrics table."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("create schema if not exists marts")
    if with_metrics:
        con.execute(
            """
            create table marts.model_metrics as
            select * from (values
                ('rv_har', 'SPY', 'oos_r2',            0.25, current_timestamp),
                ('rv_har', 'SPY', 'qlike_skill_ratio',  0.92, current_timestamp),
                ('rv_har', 'SPY', 'folds_passed',       4.0,  current_timestamp),
                ('rv_har', 'SPY', 'n_folds',            5.0,  current_timestamp),
                ('rv_har', 'SPY', 'n_obs',              300.0, current_timestamp)
            ) t(model, symbol, metric, value, trained_at)
            """
        )
    else:
        # Empty table with correct schema — simulates model not yet trained.
        con.execute(
            "create table marts.model_metrics "
            "(model varchar, symbol varchar, metric varchar, value double, trained_at timestamp)"
        )
    con.close()
    return db_path


def _connect_to(db_path):
    """Return a connect() shim that opens the fixture DB."""
    return lambda *a, **k: duckdb.connect(str(db_path), read_only=k.get("read_only", False))


# ---------------------------------------------------------------------------
# Verdict fixtures
# ---------------------------------------------------------------------------

_CLEARED_VERDICT = {
    "cleared": True,
    "reasons": [],
    "oos_r2": 0.25,
    "qlike_skill_ratio": 0.92,
    "folds_passed": 4,
    "n_folds": 5,
    "n_obs": 300,
}

_NOT_CLEARED_VERDICT = {
    "cleared": False,
    "reasons": [
        "oos_r2=0.05 < R2_MIN=0.10 — model does not beat the persistence baseline out-of-sample",
        "folds_passed=2 < ceil(0.6x5)=3 — skill not sustained across folds",
    ],
    "oos_r2": 0.05,
    "qlike_skill_ratio": 0.92,
    "folds_passed": 2,
    "n_folds": 5,
    "n_obs": 300,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ml_gate_cleared_exits_zero(monkeypatch, tmp_path, capsys):
    """When skill_verdict returns cleared=True, exit code must be 0."""
    db_path = _make_db(tmp_path, with_metrics=True)
    monkeypatch.setattr(cli, "connect", _connect_to(db_path))

    with patch("mmi.ml.skill_gate.skill_verdict", return_value=_CLEARED_VERDICT) as mock_sv:
        args = argparse.Namespace(symbol="SPY", warn_only=False)
        rc = cli.cmd_ml_gate(args)

    assert rc == 0
    mock_sv.assert_called_once()
    out = capsys.readouterr().out
    assert "CLEARED" in out
    assert "SPY" in out


def test_ml_gate_not_cleared_exits_nonzero(monkeypatch, tmp_path, capsys):
    """When skill_verdict returns cleared=False, exit code must be non-zero + reasons printed."""
    db_path = _make_db(tmp_path, with_metrics=False)
    monkeypatch.setattr(cli, "connect", _connect_to(db_path))

    with patch("mmi.ml.skill_gate.skill_verdict", return_value=_NOT_CLEARED_VERDICT) as mock_sv:
        args = argparse.Namespace(symbol="SPY", warn_only=False)
        rc = cli.cmd_ml_gate(args)

    assert rc != 0
    mock_sv.assert_called_once()
    out = capsys.readouterr().out
    assert "NOT CLEARED" in out
    # Both reasons must appear.
    for reason in _NOT_CLEARED_VERDICT["reasons"]:
        assert reason in out


def test_ml_gate_warn_only_not_cleared_exits_zero(monkeypatch, tmp_path, capsys):
    """--warn-only must exit 0 even when the gate is not cleared."""
    db_path = _make_db(tmp_path, with_metrics=False)
    monkeypatch.setattr(cli, "connect", _connect_to(db_path))

    with patch("mmi.ml.skill_gate.skill_verdict", return_value=_NOT_CLEARED_VERDICT):
        args = argparse.Namespace(symbol="SPY", warn_only=True)
        rc = cli.cmd_ml_gate(args)

    assert rc == 0
    out = capsys.readouterr().out
    # Verdict is still printed even in warn-only mode.
    assert "NOT CLEARED" in out


def test_ml_gate_missing_table_not_cleared_no_exception(monkeypatch, tmp_path):
    """When model_metrics table is absent, the command must NOT raise — returns not-cleared."""
    db_path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db_path))
    # Intentionally do NOT create the marts schema or model_metrics table.
    con.close()

    monkeypatch.setattr(cli, "connect", _connect_to(db_path))

    # Do NOT monkeypatch skill_verdict — let the real one run with an empty DataFrame
    # (the CLI must catch the CatalogException / table-not-found and pass an empty DF).
    args = argparse.Namespace(symbol="SPY", warn_only=True)
    # Must not raise regardless of the DB state; warn-only -> exit 0.
    rc = cli.cmd_ml_gate(args)
    assert rc == 0


def test_ml_gate_argparse_wiring():
    """The 'ml-gate' subcommand is registered with --symbol and --warn-only defaults."""
    parser = cli.build_parser()
    args = parser.parse_args(["ml-gate"])
    assert args.symbol == "SPY"
    assert args.warn_only is False
    assert args.func is cli.cmd_ml_gate


def test_ml_gate_argparse_custom_symbol_and_warn_only():
    """--symbol and --warn-only flags are parsed correctly."""
    parser = cli.build_parser()
    args = parser.parse_args(["ml-gate", "--symbol", "QQQ", "--warn-only"])
    assert args.symbol == "QQQ"
    assert args.warn_only is True
