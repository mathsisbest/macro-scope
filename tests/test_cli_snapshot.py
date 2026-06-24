"""`mmi snapshot` exports every marts table to Parquet — the public demo's static, secret-free
data source. Exporting the whole schema means new marts are picked up automatically.

Also covers the MMI_PORTFOLIO_N_BOOT env knob wired into cmd_portfolio (task D1).
"""

import argparse
import contextlib
import os
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pandas as pd

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


# ---------------------------------------------------------------------------
# MMI_PORTFOLIO_N_BOOT env knob (task D1)
# ---------------------------------------------------------------------------


def _minimal_asset_daily(window_id: str = "ex_btc_2002") -> pd.DataFrame:
    """50 rows with a single strategy — enough for bootstrap_strategy_stats (needs ≥2)."""
    dates = pd.bdate_range("2020-01-01", periods=50)
    return pd.DataFrame(
        {
            "window_id": window_id,
            "strategy": "equal_weight",
            "date": dates,
            "daily_return": np.random.default_rng(0).normal(0.0003, 0.01, 50),
        }
    )


def _make_portfolio_mocks(captured: dict):
    """Return a dict of patches that short-circuit the heavy compute pipeline.

    ``captured`` is mutated in-place: ``captured['n_boot_stats']`` gets the
    n_boot value passed to bootstrap_strategy_stats.
    """
    minimal = _minimal_asset_daily()

    import mmi.portfolio.stats as _stats

    real_bootstrap = _stats.bootstrap_strategy_stats

    def fake_bootstrap(returns_long, *, window="ex_btc_2002", n_boot=2000, **kw):
        captured["n_boot_stats"] = n_boot
        # Delegate to the real implementation with n_boot so the return shape is correct.
        return real_bootstrap(returns_long, window=window, n_boot=n_boot, **kw)

    real_btc_effect = _stats.paired_btc_effect

    def fake_btc_effect(ex_df, inc_df, *, n_boot=2000, **kw):
        captured["n_boot_btc"] = n_boot
        return real_btc_effect(ex_df, inc_df, n_boot=n_boot, **kw)

    compute_patches = {
        "mmi.portfolio.compute.compute_ml_mu_panel": MagicMock(
            return_value=(pd.DataFrame(), pd.DataFrame())
        ),
        "mmi.portfolio.compute.compute_portfolio_returns": MagicMock(return_value=minimal),
        "mmi.portfolio.compute.compute_attribution": MagicMock(
            return_value=pd.DataFrame(columns=["window_id", "strategy", "symbol"])
        ),
        "mmi.portfolio.compute.btc_aligned_returns": MagicMock(
            return_value=pd.DataFrame({"date": [], "daily_return": []})
        ),
        "mmi.portfolio.compute.window_asset_daily": MagicMock(return_value=minimal),
        "mmi.portfolio.stats.bootstrap_strategy_stats": fake_bootstrap,
        "mmi.portfolio.stats.paired_btc_effect": fake_btc_effect,
        "mmi.ingestion.loader.reset_portfolio_raw_tables": MagicMock(),
        "mmi.settings.load_assets": MagicMock(return_value={}),
    }
    return compute_patches


def _run_cmd_portfolio_with_nboot(monkeypatch, tmp_path, n_boot_env: str | None) -> dict:
    """Run cmd_portfolio with heavy compute mocked; return captured dict."""
    db = tmp_path / "p.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists raw")
    con.execute("create schema if not exists marts")
    # fct_asset_daily must exist so cmd_portfolio's SELECT succeeds.
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        "('SPY', DATE '2020-01-01', 0.001, 'equities')) "
        "t(symbol, date, daily_return, asset_class)"
    )
    con.close()

    captured: dict = {}
    patches = _make_portfolio_mocks(captured)

    env_ctx = {"MMI_PORTFOLIO_N_BOOT": n_boot_env} if n_boot_env is not None else {}
    # Strip the env var when testing the default so a stale env from a parent process can't bleed.
    env_strip = n_boot_env is None

    with patch.dict(os.environ, env_ctx, clear=False):
        if env_strip:
            os.environ.pop("MMI_PORTFOLIO_N_BOOT", None)
        ctx_managers = [patch(k, v) for k, v in patches.items()]
        # Also patch connect() so cmd_portfolio opens our fixture DB.
        ctx_managers.append(patch.object(cli, "connect", _connect_to(db)))
        for cm in ctx_managers:
            cm.__enter__()
        try:
            cli.cmd_portfolio(argparse.Namespace())
        finally:
            for cm in reversed(ctx_managers):
                with contextlib.suppress(Exception):
                    cm.__exit__(None, None, None)

    return captured


def test_portfolio_nboot_knob_env_var_is_honoured(monkeypatch, tmp_path):
    """MMI_PORTFOLIO_N_BOOT is read and passed to bootstrap_strategy_stats."""
    captured = _run_cmd_portfolio_with_nboot(monkeypatch, tmp_path, "42")
    assert captured.get("n_boot_stats") == 42, (
        f"Expected n_boot=42 from env, got {captured.get('n_boot_stats')}"
    )


def test_portfolio_nboot_knob_default_is_2000(monkeypatch, tmp_path):
    """When MMI_PORTFOLIO_N_BOOT is unset the default is 2000."""
    captured = _run_cmd_portfolio_with_nboot(monkeypatch, tmp_path, None)
    assert captured.get("n_boot_stats") == 2000, (
        f"Expected default n_boot=2000, got {captured.get('n_boot_stats')}"
    )
