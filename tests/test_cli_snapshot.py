"""`mmi snapshot` exports every marts table to Parquet — the public demo's static, secret-free
data source. Exporting the whole schema means new marts are picked up automatically."""

import argparse

import duckdb

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
        "create table marts.fct_portfolio_returns as select * from (values "
        "('ex_btc_2002', 'equal_weight', DATE '2020-01-01', 0.01)) "
        "t(window_id, strategy, date, daily_return)"
    )
    con.close()


def test_snapshot_exports_every_marts_table(monkeypatch, tmp_path):
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))

    assert cli.cmd_snapshot(argparse.Namespace()) == 0

    # one Parquet per marts table, readable, with the right rows
    assert {p.name for p in out.glob("*.parquet")} == {
        "dim_asset.parquet",
        "fct_portfolio_returns.parquet",
    }
    rt = duckdb.connect()
    try:
        n = rt.execute(f"select count(*) from '{out / 'dim_asset.parquet'}'").fetchone()[0]
    finally:
        rt.close()
    assert n == 2  # round-trips


def test_snapshot_with_no_marts_is_a_noop(monkeypatch, tmp_path):
    db = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists marts")
    con.close()
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))

    assert cli.cmd_snapshot(argparse.Namespace()) == 0
    assert not list(out.glob("*.parquet"))  # nothing exported, no crash
