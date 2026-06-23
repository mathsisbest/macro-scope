"""Snapshot-read mode: the dashboard reads the committed Parquet snapshot IN-PROCESS — no live
DuckDB/MotherDuck, no secrets. Round-trips marts -> `mmi snapshot` -> Parquet -> accessors, and
asserts the live DB connector (`connect`) is never called."""

import argparse

import duckdb
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
