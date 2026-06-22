"""build_returns_panel pivots correctly; compute covers all strategies; the CLI lands the table."""

import argparse

import duckdb
import numpy as np
import pandas as pd

import mmi.cli as cli
from mmi.portfolio.compute import build_returns_panel, compute_portfolio_returns


def _long(n_days: int = 120, assets: tuple = ("A", "B", "C"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    for asset in assets:
        rets = rng.normal(0.0004, 0.01, n_days)
        for day, ret in zip(idx, rets, strict=True):
            rows.append({"symbol": asset, "date": day, "daily_return": ret})
    return pd.DataFrame(rows)


def test_build_returns_panel_pivots_to_wide():
    panel = build_returns_panel(_long(10, ("A", "B")))
    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == 10
    assert panel.index.is_monotonic_increasing


def test_compute_portfolio_returns_covers_all_strategies():
    out = compute_portfolio_returns(_long(150), lookback=30, freq="M", cost=0.001)
    assert set(out["strategy"].unique()) == {"equal_weight", "inverse_vol", "risk_parity"}
    assert list(out.columns) == ["strategy", "date", "daily_return", "cumulative_return"]


def test_cmd_portfolio_lands_raw_portfolio_returns(monkeypatch, tmp_path):
    db = tmp_path / "p.duckdb"
    setup = duckdb.connect(str(db))
    setup.execute("create schema if not exists marts")
    setup.register("_fad", _long(320))
    setup.execute("create table marts.fct_asset_daily as select * from _fad")
    setup.close()

    monkeypatch.setattr(cli, "connect", lambda *a, **k: duckdb.connect(str(db)))
    assert cli.cmd_portfolio(argparse.Namespace()) == 0

    check = duckdb.connect(str(db))
    try:
        strategies = check.execute("select distinct strategy from raw.portfolio_returns").fetchall()
    finally:
        check.close()
    assert {s[0] for s in strategies} == {"equal_weight", "inverse_vol", "risk_parity"}
