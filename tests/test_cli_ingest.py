"""cmd_ingest: a required-source failure fails the run; an optional one does not — both audited."""

import argparse

import duckdb
import pandas as pd

import mmi.cli as cli
import mmi.ingestion as ingestion
from mmi.ingestion.base import Extractor


class _FailRequired(Extractor):
    source = "fail_required"
    table = "raw.fail_required"
    keys = ["a"]
    required = True

    def fetch(self) -> pd.DataFrame:
        raise RuntimeError("required source down")


class _FailOptional(Extractor):
    source = "fail_optional"
    table = "raw.fail_optional"
    keys = ["a"]
    required = False

    def fetch(self) -> pd.DataFrame:
        raise RuntimeError("optional source down")


class _FailWithSecret(Extractor):
    source = "fail_secret"
    table = "raw.fail_secret"
    keys = ["a"]
    required = True

    def fetch(self) -> pd.DataFrame:
        raise RuntimeError(
            "400 Bad Request for url "
            "'https://api.stlouisfed.org/fred/series/observations?series_id=DGS10"
            "&api_key=SECRETXYZ123&file_type=json'"
        )


def _wire(monkeypatch, tmp_path, extractors):
    db = tmp_path / "ingest.duckdb"
    monkeypatch.setattr(cli, "connect", lambda *a, **k: duckdb.connect(str(db)))
    monkeypatch.setattr(ingestion, "EXTRACTORS", extractors)
    return db


def _statuses(db):
    con = duckdb.connect(str(db))
    try:
        return dict(con.execute("select source, status from raw.pipeline_runs").fetchall())
    finally:
        con.close()


def test_required_failure_returns_nonzero_and_is_audited(monkeypatch, tmp_path):
    db = _wire(monkeypatch, tmp_path, [_FailRequired])
    assert cli.cmd_ingest(argparse.Namespace()) == 1
    assert _statuses(db)["fail_required"] == "failed"


def test_optional_failure_returns_zero_but_is_audited(monkeypatch, tmp_path):
    db = _wire(monkeypatch, tmp_path, [_FailOptional])
    assert cli.cmd_ingest(argparse.Namespace()) == 0
    assert _statuses(db)["fail_optional"] == "failed"


def test_failure_message_is_redacted_in_audit(monkeypatch, tmp_path):
    db = _wire(monkeypatch, tmp_path, [_FailWithSecret])
    cli.cmd_ingest(argparse.Namespace())
    con = duckdb.connect(str(db))
    try:
        msg = con.execute(
            "select message from raw.pipeline_runs where source = 'fail_secret'"
        ).fetchone()[0]
    finally:
        con.close()
    assert "SECRETXYZ123" not in msg  # the key never reaches the audit table
    assert "api_key=***" in msg
