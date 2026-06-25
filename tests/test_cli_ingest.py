"""cmd_ingest classification & audit:

- a *required*-source failure fails the run (exit 1); an *optional* one does not (exit 0);
- a key-gated source with its key absent is *skipped* (audited "skipped", fetch never called) so a
  keyless run still lands the no-key core and exits 0;
all three are recorded in raw.pipeline_runs.
"""

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


class _NeedsKey(Extractor):
    """A key-gated source: when its key is absent it skips (audited 'skipped'), without fetching."""

    source = "needs_key"
    table = "raw.needs_key"
    keys = ["a"]
    required = True
    _fetched = False
    _reason: str | None = "NEEDS_KEY not set"

    def skip_reason(self) -> str | None:
        return self._reason

    def fetch(self) -> pd.DataFrame:
        type(self)._fetched = True
        return pd.DataFrame({"a": [1]})


class _Keyless(Extractor):
    """A no-key source (e.g. Yahoo / World Bank) that always lands rows."""

    source = "keyless"
    table = "raw.keyless"
    keys = ["a"]

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2]})


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


def test_keyed_source_skips_when_unconfigured(monkeypatch, tmp_path):
    _NeedsKey._fetched = False
    _NeedsKey._reason = "NEEDS_KEY not set"
    db = _wire(monkeypatch, tmp_path, [_NeedsKey])
    assert cli.cmd_ingest(argparse.Namespace()) == 0  # a skip is not a failure
    assert _statuses(db)["needs_key"] == "skipped"  # audited as skipped, not failed/success
    assert _NeedsKey._fetched is False  # fetch never attempted


def test_keyed_source_runs_when_configured(monkeypatch, tmp_path):
    _NeedsKey._fetched = False
    _NeedsKey._reason = None  # key present -> no skip
    db = _wire(monkeypatch, tmp_path, [_NeedsKey])
    assert cli.cmd_ingest(argparse.Namespace()) == 0
    assert _statuses(db)["needs_key"] == "success"
    assert _NeedsKey._fetched is True


def test_unkeyed_run_still_lands_keyless_core_and_exits_zero(monkeypatch, tmp_path):
    """The #51 acceptance: unkeyed, the key-gated source skips and the keyless core still lands."""
    _NeedsKey._fetched = False
    _NeedsKey._reason = "NEEDS_KEY not set"
    db = _wire(monkeypatch, tmp_path, [_NeedsKey, _Keyless])
    assert cli.cmd_ingest(argparse.Namespace()) == 0
    statuses = _statuses(db)
    assert statuses["needs_key"] == "skipped"
    assert statuses["keyless"] == "success"


def test_fred_skips_only_when_unkeyed(monkeypatch):
    """FRED skips when its key is absent and runs when present; keyed failures stay fatal."""
    import mmi.ingestion.fred as fred_mod

    monkeypatch.setattr(fred_mod.settings, "fred_api_key", "")
    assert fred_mod.FredExtractor(loader=None).skip_reason() is not None
    monkeypatch.setattr(fred_mod.settings, "fred_api_key", "a-key")
    assert fred_mod.FredExtractor(loader=None).skip_reason() is None


# ---------------------------------------------------------------------------
# G1 — base.run() audit-mask fix (#50 item 2)
# ---------------------------------------------------------------------------


class _FailFetchAuditRaises(Extractor):
    """Extractor whose fetch() fails AND whose loader.finish_run() also raises.

    The ORIGINAL fetch exception must propagate; the audit failure must never
    replace/mask it.
    """

    source = "fail_fetch_audit_raises"
    table = "raw.fail_fetch_audit_raises"
    keys = ["a"]
    required = True

    def fetch(self) -> pd.DataFrame:
        raise RuntimeError("original ingest error — must propagate")


class _BrokenAuditLoader:
    """Minimal loader stub where finish_run raises to simulate an audit write failure."""

    def __init__(self) -> None:
        self._run_id = 0

    def start_run(self, source: str) -> int:
        self._run_id += 1
        return self._run_id

    def finish_run(self, run_id: int, rows: int, status: str, message: str = "") -> None:
        if status == "failed":
            raise OSError("audit write failed — DB locked")

    def upsert(self, table: str, df: pd.DataFrame, keys: list) -> int:  # type: ignore[type-arg]
        return len(df)


def test_audit_write_failure_does_not_mask_original_ingest_exception():
    """G1: when finish_run raises inside the except block the ORIGINAL exception propagates."""
    loader = _BrokenAuditLoader()
    extractor = _FailFetchAuditRaises(loader)

    import pytest

    with pytest.raises(RuntimeError, match="original ingest error — must propagate"):
        extractor.run()


def test_audit_write_failure_does_not_mask_original_exception_type():
    """G1: the raised type is exactly the original (RuntimeError), not the audit OSError."""
    import pytest

    loader = _BrokenAuditLoader()
    extractor = _FailFetchAuditRaises(loader)

    with pytest.raises(RuntimeError) as exc_info:
        extractor.run()

    # Ensure we are NOT seeing the audit OSError escape as the live exception
    assert "audit write failed" not in str(exc_info.value)
