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


def test_coingecko_is_optional_so_a_keyless_run_stays_green():
    """CoinGecko works keyless but is non-fatal, so a scheduled ingest can't fail on it."""
    from mmi.ingestion.coingecko import CoinGeckoExtractor

    assert CoinGeckoExtractor.required is False
    assert CoinGeckoExtractor(loader=None).skip_reason() is None  # never skips: it works keyless
