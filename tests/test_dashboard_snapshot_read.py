"""Snapshot-read mode: the dashboard reads the committed Parquet snapshot IN-PROCESS — no live
DuckDB/MotherDuck, no secrets. Round-trips marts -> `mmi snapshot` -> Parquet -> accessors, and
asserts the live DB connector (`connect`) is never called."""

import argparse
from datetime import datetime

import duckdb
import pandas as pd
from dashboard import data

import mmi.cli as cli
import mmi.settings as settings_mod


def _connect_to(db):
    # Mirror connect(read_only=...) so cmd_snapshot's read-only open works against the fixture.
    return lambda *a, **k: duckdb.connect(str(db), read_only=k.get("read_only", False))


def _marts_db(path) -> None:
    con = duckdb.connect(str(path))
    con.execute("create schema if not exists marts")
    con.execute(
        "create table marts.dim_asset as "
        "select * from (values ('SPY', 'equities'), ('BTC', 'crypto')) t(symbol, asset_class)"
    )
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        "('SPY', DATE '2024-01-02', 470.0, 0.012, 9.5, 465.0)) "
        "t(symbol, date, close, daily_return, vol_20d, ma_50)"
    )
    con.close()


def _export_snapshot(monkeypatch, tmp_path):
    """Build a marts DB and export it to a Parquet snapshot dir; return the dir."""
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))
    assert cli.cmd_snapshot(argparse.Namespace()) == 0
    return out


def test_dashboard_reads_snapshot_without_touching_a_db(monkeypatch, tmp_path):
    _export_snapshot(monkeypatch, tmp_path)
    monkeypatch.setattr(settings_mod.settings, "snapshot_mode", True)

    # Any call to the live DB connector is a failure: snapshot mode must read Parquet in-process.
    def _no_db(*a, **k):
        raise AssertionError("connect() called in snapshot mode — must read Parquet, not a DB")

    monkeypatch.setattr(data, "connect", _no_db)
    data.query.clear()  # st.cache_data caches by (sql, params); mode isn't part of the key

    assert data.db_exists() is True  # Parquet files present in snapshot_dir

    assets = data.assets()
    assert set(assets["symbol"]) == {"SPY", "BTC"}
    assert set(assets.columns) == {"symbol", "asset_class"}

    daily = data.asset_daily("SPY")
    assert not daily.empty
    assert {"date", "close", "daily_return", "vol_20d", "ma_50"} <= set(daily.columns)

    # A mart with no Parquet file (ML didn't run) degrades to empty, not a crash — like a missing
    # table on a live DB. raw.pipeline_runs is likewise never snapshotted.
    assert data.ml_forecast().empty
    assert data.pipeline_runs().empty


def test_db_exists_false_when_snapshot_dir_has_no_parquet(monkeypatch, tmp_path):
    empty = tmp_path / "public"
    empty.mkdir()
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", empty)
    monkeypatch.setattr(settings_mod.settings, "snapshot_mode", True)
    assert data.db_exists() is False


def test_storage_label_reflects_snapshot_mode(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "snapshot_mode", True)
    label = settings_mod.settings.storage_label()
    assert label.startswith("Parquet snapshot")
    # the snapshot label must never imply a live/MotherDuck backend
    assert "MotherDuck" not in label and "DuckDB" not in label


# ---------------------------------------------------------------------------
# H2 — brief snapshot round-trip
# ---------------------------------------------------------------------------


def _seed_and_snapshot(monkeypatch, tmp_path):
    """Run ``mmi seed`` (writes an offline brief) then ``mmi snapshot`` into a temp dir.

    Returns the snapshot dir Path.  Uses isolated DB path and snapshot dir so the test
    never touches the real DB or data/public.
    """
    db_path = tmp_path / "seed.duckdb"
    snap_dir = tmp_path / "public"

    # Route the DB to our isolated temp file.
    monkeypatch.setattr(settings_mod.settings, "duckdb_path", db_path)
    # Route the snapshot export to our temp dir.
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", snap_dir)
    # Ensure snapshot reads hit our dir (set before cmd_snapshot patches cli.connect).
    # Patch cli.connect so cmd_snapshot opens the same temp DB (read_only=True).
    monkeypatch.setattr(
        cli,
        "connect",
        lambda *a, **k: duckdb.connect(str(db_path), read_only=k.get("read_only", False)),
    )

    assert cli.cmd_seed(argparse.Namespace()) == 0, "cmd_seed must exit 0"
    assert cli.cmd_snapshot(argparse.Namespace()) == 0, "cmd_snapshot must exit 0"

    return snap_dir


def test_brief_snapshot_roundtrip_returns_one_row(monkeypatch, tmp_path):
    """After seed->snapshot, latest_brief() must return exactly one row in snapshot mode.

    Checks:
    - exactly one row (not zero, not two)
    - engine column is present and non-empty (seed writes 'offline-template')
    - brief column is present and non-empty
    - created_at is parseable as a datetime (Contract G: wall-clock generation time)
    """
    snap_dir = _seed_and_snapshot(monkeypatch, tmp_path)

    # Switch data.py into snapshot mode pointing at our temp dir.
    monkeypatch.setattr(settings_mod.settings, "snapshot_mode", True)
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", snap_dir)
    # Invalidate any cached query results from prior tests.
    data.query.clear()

    result = data.latest_brief()

    assert not result.empty, "latest_brief() must return a row after seed->snapshot"
    assert len(result) == 1, f"Expected exactly 1 brief row, got {len(result)}"

    row = result.iloc[0]

    # engine must be present and non-empty — seed writes 'offline-template'
    assert "engine" in result.columns, "market_brief must have an 'engine' column"
    assert row["engine"], "engine must be non-empty"

    # brief body must be present and non-empty
    assert "brief" in result.columns, "market_brief must have a 'brief' column"
    assert row["brief"], "brief body must be non-empty"

    # created_at must be parseable as a datetime (Contract G: wall-clock stamp)
    assert "created_at" in result.columns, "market_brief must have a 'created_at' column"
    created_at = row["created_at"]
    # DuckDB may return a Timestamp, datetime, or ISO string — all must be parseable.
    if isinstance(created_at, str):
        parsed = datetime.fromisoformat(created_at)
    else:
        # pandas Timestamp or datetime.datetime — both are datetime-like
        parsed = pd.Timestamp(created_at).to_pydatetime()
    assert isinstance(parsed, datetime), (
        f"created_at must be a parseable datetime, got {created_at!r}"
    )


def test_brief_snapshot_missing_parquet_returns_empty_not_crash(monkeypatch, tmp_path):
    """If market_brief.parquet is deleted from the snapshot dir, latest_brief() must return an
    empty DataFrame — NOT raise an exception.

    Regression guard for Contract B: query() swallows CatalogException (missing table/view)
    and returns an empty frame; the caller checks for emptiness via 'No brief yet' messaging.
    """
    snap_dir = _seed_and_snapshot(monkeypatch, tmp_path)

    # Delete the brief Parquet so the snapshot dir has no market_brief view.
    brief_parquet = snap_dir / "market_brief.parquet"
    assert brief_parquet.exists(), (
        "market_brief.parquet must exist after seed->snapshot for this test to be meaningful"
    )
    brief_parquet.unlink()

    monkeypatch.setattr(settings_mod.settings, "snapshot_mode", True)
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", snap_dir)
    data.query.clear()

    # Must not raise — CatalogException is swallowed to an empty frame by query().
    result = data.latest_brief()

    assert result.empty, (
        "latest_brief() must return an empty DataFrame when market_brief.parquet is absent"
    )
