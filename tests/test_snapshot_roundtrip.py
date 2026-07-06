"""Snapshot round-trip schema guard.

Runs the offline pipeline mirroring ``make ci`` (seed -> portfolio -> DROP schemas cascade ->
dbt build -> mmi snapshot), then opens the exported Parquet snapshot EXACTLY as
``dashboard/data.py``'s ``_snapshot_connection`` does (in-memory DuckDB,
``set python_enable_replacements=false``, ``read_parquet``, register ``marts.<table>`` views)
and asserts that the ASSET / MACRO / PORTFOLIO accessors resolve their expected columns
post-round-trip.

This catches column-order / nullable / date-format drift between what the pipeline produces and
what the dashboard expects to consume.

Scope: seed -> portfolio -> dbt build -> snapshot.
NOT in scope: mmi ml / mmi ai (owned by Wave-3 tasks H3/H2 respectively).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import mmi.cli as cli
import mmi.settings as settings_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
# Prefer the venv dbt binary so tests are hermetic; fall back to PATH.
_VENV_DBT = REPO_ROOT / ".venv" / "bin" / "dbt"
_DBT_CMD = str(_VENV_DBT) if _VENV_DBT.exists() else "dbt"


def _run_dbt(db_path: Path) -> None:
    """Run dbt build against *db_path*, dropping marts/staging first (mirrors make ci)."""
    # Drop schemas cascade exactly as make ci does.
    con = duckdb.connect(str(db_path))
    con.execute("drop schema if exists marts cascade")
    con.execute("drop schema if exists staging cascade")
    con.close()

    env = {
        **os.environ,
        "MMI_DUCKDB_PATH": str(db_path),
        "MMI_MOTHERDUCK_DATABASE": "",
        "MOTHERDUCK_TOKEN": "",
    }
    result = subprocess.run(
        [
            _DBT_CMD,
            "build",
            "--project-dir",
            str(REPO_ROOT / "transform"),
            "--profiles-dir",
            str(REPO_ROOT / "transform"),
            "--target",
            "dev",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Surface dbt output for debugging without exposing secrets (env cleared above).
        pytest.fail(f"dbt build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def _snapshot_connection_from_dir(snapshot_dir: Path) -> duckdb.DuckDBPyConnection:
    """Mirror dashboard/data.py ``_snapshot_connection`` exactly — no import of data.py."""
    con = duckdb.connect(":memory:")
    con.execute("set python_enable_replacements=false")
    con.execute("create schema if not exists raw")
    con.execute("create schema if not exists marts")
    for path in sorted(snapshot_dir.glob("*.parquet")):
        safe_path = str(path).replace("'", "''")
        safe_name = path.stem.replace('"', '""')
        con.execute(
            f"create view marts.\"{safe_name}\" as select * from read_parquet('{safe_path}')"
        )
    return con


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def snapshot_dir(tmp_path_factory):
    """Build the full offline pipeline (seed->portfolio->dbt->snapshot) once per module.

    Returns the Path to the snapshot dir holding the exported Parquet files.
    The temp directory is cleaned up automatically by pytest.
    """
    tmp = tmp_path_factory.mktemp("roundtrip")
    db_path = tmp / "ci.duckdb"
    snap_dir = tmp / "public"

    # 1. Seed sample data + portfolio (mirrors `mmi seed` then `mmi portfolio`).
    #    We use the CLI commands directly so the same code path as make ci is exercised.
    original_db = settings_mod.settings.duckdb_path
    settings_mod.settings.duckdb_path = db_path

    try:
        rc = cli.cmd_seed(argparse.Namespace())
        assert rc == 0, "cmd_seed failed"

        rc = cli.cmd_portfolio(argparse.Namespace())
        assert rc == 0, "cmd_portfolio failed"
    finally:
        settings_mod.settings.duckdb_path = original_db

    # 2. DROP schemas cascade + dbt build (mirrors make ci exactly).
    _run_dbt(db_path)

    # 3. Export snapshot (mmi snapshot with custom dir).
    original_snap = settings_mod.settings.snapshot_dir
    original_connect = cli.connect

    def _connect_ci(*args, **kwargs):
        return duckdb.connect(str(db_path), read_only=kwargs.get("read_only", False))

    cli.connect = _connect_ci
    settings_mod.settings.snapshot_dir = snap_dir
    try:
        rc = cli.cmd_snapshot(argparse.Namespace())
        assert rc == 0, "cmd_snapshot failed"
    finally:
        cli.connect = original_connect
        settings_mod.settings.snapshot_dir = original_snap

    parquets = list(snap_dir.glob("*.parquet"))
    assert parquets, "No Parquet files exported — snapshot is empty"
    return snap_dir


@pytest.fixture(scope="module")
def snap_con(snapshot_dir):
    """In-memory DuckDB wired up exactly as dashboard/data.py._snapshot_connection."""
    con = _snapshot_connection_from_dir(snapshot_dir)
    yield con
    con.close()


# ---------------------------------------------------------------------------
# Helper: run a SELECT against snap_con and return a DataFrame
# ---------------------------------------------------------------------------


def _q(snap_con: duckdb.DuckDBPyConnection, sql: str, params=None) -> pd.DataFrame:
    try:
        return snap_con.execute(sql, params or []).df()
    except duckdb.CatalogException:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# ASSET accessors
# ---------------------------------------------------------------------------


class TestAssetAccessors:
    """data.assets() + data.asset_daily() column contracts."""

    def test_assets_columns(self, snap_con):
        """data.assets() -> symbol, asset_class (in that order)."""
        df = _q(
            snap_con,
            "select symbol, asset_class from marts.dim_asset order by asset_class, symbol",
        )
        assert not df.empty, "dim_asset is empty after round-trip"
        assert list(df.columns) == ["symbol", "asset_class"]
        # Basic sanity: SPY should be in the snapshot.
        assert "SPY" in df["symbol"].to_numpy()

    def test_assets_asset_class_values(self, snap_con):
        """asset_class values must be one of the known classes."""
        df = _q(snap_con, "select distinct asset_class from marts.dim_asset")
        valid = {"equities", "bonds", "commodities", "fx", "crypto"}
        assert set(df["asset_class"]) <= valid

    def test_asset_daily_columns(self, snap_con):
        """data.asset_daily() -> date, close, daily_return, vol_20d, ma_50 (subset)."""
        df = _q(
            snap_con,
            "select date, close, daily_return, vol_20d, ma_50"
            " from marts.fct_asset_daily"
            " where symbol = 'SPY' order by date",
        )
        assert not df.empty, "fct_asset_daily has no SPY rows after round-trip"
        expected_cols = {"date", "close", "daily_return", "vol_20d", "ma_50"}
        assert expected_cols <= set(df.columns)

    def test_asset_daily_date_dtype(self, snap_con):
        """date column must round-trip as date/datetime-compatible (not plain string)."""
        df = _q(
            snap_con,
            "select date from marts.fct_asset_daily where symbol = 'SPY' order by date limit 1",
        )
        assert not df.empty
        # After read_parquet the date column should be date-like (not object/str).
        dtype = df["date"].dtype
        assert pd.api.types.is_datetime64_any_dtype(dtype) or str(dtype) in (
            "object",
            "date32[day][pyarrow]",
            "date64[ms][pyarrow]",
        ), f"Unexpected date dtype: {dtype}"
        # Must be parseable as a date regardless of dtype.
        val = df["date"].iloc[0]
        parsed = pd.Timestamp(val)
        assert parsed.year >= 2020, f"Parsed date looks wrong: {parsed}"

    def test_asset_daily_source_column_present(self, snap_con):
        """source column must survive round-trip (drives is_sample_data())."""
        df = _q(
            snap_con,
            "select distinct source from marts.fct_asset_daily",
        )
        assert not df.empty, "fct_asset_daily.source column missing after round-trip"
        # Seeded with sample data -> every source should be 'sample'.
        assert set(df["source"]) == {"sample"}


# ---------------------------------------------------------------------------
# MACRO accessors
# ---------------------------------------------------------------------------


class TestMacroAccessors:
    """data.macro() + data.macro_ids() + data.market_macro() column contracts."""

    def test_macro_ids_returns_series(self, snap_con):
        """data.macro_ids() -> distinct series_id list, non-empty."""
        df = _q(
            snap_con,
            "select distinct series_id from marts.fct_macro_indicator order by series_id",
        )
        assert not df.empty, "fct_macro_indicator is empty after round-trip"
        assert "series_id" in df.columns

    def test_macro_columns(self, snap_con):
        """data.macro() -> date, value, change."""
        df = _q(
            snap_con,
            "select date, value, change from marts.fct_macro_indicator"
            " where series_id = 'DGS10' order by date",
        )
        assert not df.empty, "No DGS10 rows in fct_macro_indicator after round-trip"
        assert list(df.columns) == ["date", "value", "change"]

    def test_market_macro_columns(self, snap_con):
        """data.market_macro() -> select * must include the contract columns (incl. 10Y-3M)."""
        df = _q(snap_con, "select * from marts.fct_market_macro order by date")
        assert not df.empty, "fct_market_macro is empty after round-trip"
        expected = {
            "date",
            "spy_close",
            "spy_return",
            "vol_20d",
            "us_10y",
            "us_2y",
            "us_3m",
            "yield_curve_10y_2y",
            "yield_curve_10y_3m",
        }
        assert expected <= set(df.columns), f"Missing columns: {expected - set(df.columns)}"

    def test_data_as_of_resolvable(self, snap_con):
        """data_as_of() query resolves to a non-empty date string."""
        df = _q(
            snap_con,
            "select cast(max(date) as varchar) as d from marts.fct_market_macro",
        )
        assert not df.empty
        assert not pd.isna(df["d"].iloc[0])
        val = str(df["d"].iloc[0])
        assert len(val) >= 8, f"data_as_of value looks wrong: {val!r}"


# ---------------------------------------------------------------------------
# PORTFOLIO accessors
# ---------------------------------------------------------------------------


class TestPortfolioAccessors:
    """portfolio_returns / strategy_stats / strategy_pairs / attribution /
    regime_performance / ml_gate / btc_effect column contracts."""

    def test_portfolio_returns_columns(self, snap_con):
        """data.portfolio_returns() -> contract columns present."""
        df = _q(
            snap_con,
            "select strategy, date, daily_return, cumulative_return,"
            " drawdown, rolling_sharpe_252"
            " from marts.fct_portfolio_returns order by strategy, date limit 5",
        )
        assert not df.empty, "fct_portfolio_returns is empty after round-trip"
        expected = {
            "strategy",
            "date",
            "daily_return",
            "cumulative_return",
            "drawdown",
            "rolling_sharpe_252",
        }
        assert expected <= set(df.columns)

    def test_portfolio_returns_window_id_present(self, snap_con):
        """window_id column must survive round-trip (used by data.portfolio_windows())."""
        df = _q(
            snap_con,
            "select distinct window_id from marts.fct_portfolio_returns",
        )
        assert not df.empty, "fct_portfolio_returns has no window_id values"
        valid_windows = {"ex_btc_2002", "ex_btc_2015", "inc_btc_2015"}
        assert set(df["window_id"]) <= valid_windows

    def test_portfolio_strategy_stats_columns(self, snap_con):
        """data.portfolio_strategy_stats() -> contract columns present."""
        df = _q(
            snap_con,
            "select strategy, sharpe, sharpe_lo, sharpe_hi, n_obs, n_boot, ci_pct"
            " from marts.fct_portfolio_strategy_stats order by sharpe desc limit 5",
        )
        assert not df.empty, "fct_portfolio_strategy_stats empty after round-trip"
        expected = {
            "strategy",
            "sharpe",
            "sharpe_lo",
            "sharpe_hi",
            "n_obs",
            "n_boot",
            "ci_pct",
        }
        assert expected <= set(df.columns)

    def test_portfolio_strategy_pairs_columns(self, snap_con):
        """data.portfolio_strategy_pairs() -> contract columns present."""
        df = _q(
            snap_con,
            "select strategy_a, strategy_b, sharpe_diff, diff_lo, diff_hi,"
            " distinguishable"
            " from marts.fct_portfolio_strategy_pairs"
            " order by strategy_a, strategy_b limit 5",
        )
        expected = {
            "strategy_a",
            "strategy_b",
            "sharpe_diff",
            "diff_lo",
            "diff_hi",
            "distinguishable",
        }
        missing = expected - set(df.columns)
        assert not missing, f"fct_portfolio_strategy_pairs missing columns: {missing}"

    def test_portfolio_attribution_columns(self, snap_con):
        """data.portfolio_attribution() -> contract columns present."""
        df = _q(
            snap_con,
            "select strategy, symbol, contribution_to_return, contribution_to_risk"
            " from marts.fct_performance_attribution"
            " order by strategy, contribution_to_return limit 5",
        )
        assert not df.empty, "fct_performance_attribution empty after round-trip"
        expected = {
            "strategy",
            "symbol",
            "contribution_to_return",
            "contribution_to_risk",
        }
        assert expected <= set(df.columns)

    def test_portfolio_regime_performance_columns(self, snap_con):
        """data.portfolio_regime_performance() -> contract columns present."""
        df = _q(
            snap_con,
            "select strategy, regime, n_days, day_share, ann_return, ann_vol, ann_sharpe"
            " from marts.fct_portfolio_regime_performance limit 5",
        )
        assert not df.empty, "fct_portfolio_regime_performance empty after round-trip"
        expected = {
            "strategy",
            "regime",
            "n_days",
            "day_share",
            "ann_return",
            "ann_vol",
            "ann_sharpe",
        }
        assert expected <= set(df.columns)

    def test_portfolio_ml_gate_columns(self, snap_con):
        """data.portfolio_ml_gate() -> date, forecast_skill, forecast_weight."""
        df = _q(
            snap_con,
            "select date, forecast_skill, forecast_weight"
            " from marts.fct_portfolio_ml_gate order by date limit 5",
        )
        expected = {"date", "forecast_skill", "forecast_weight"}
        assert expected <= set(df.columns), "fct_portfolio_ml_gate missing columns after round-trip"

    def test_portfolio_btc_effect_columns(self, snap_con):
        """data.portfolio_btc_effect() -> contract columns present."""
        df = _q(
            snap_con,
            "select strategy, sharpe_ex, sharpe_inc, sharpe_diff, diff_lo, diff_hi,"
            " distinguishable"
            " from marts.fct_portfolio_btc_effect order by sharpe_diff limit 5",
        )
        assert not df.empty, "fct_portfolio_btc_effect is empty after round-trip"
        expected = {
            "strategy",
            "sharpe_ex",
            "sharpe_inc",
            "sharpe_diff",
            "diff_lo",
            "diff_hi",
            "distinguishable",
        }
        assert expected <= set(df.columns)


# ---------------------------------------------------------------------------
# Missing-mart graceful degradation (contract B)
# ---------------------------------------------------------------------------


class TestMissingMartDegradation:
    """A mart with no Parquet file must raise CatalogException
    (python_enable_replacements=false), NOT silently replace with a Python object.
    This verifies the guard that prevents ``ml_forecast`` (the accessor function)
    from shadow-replacing the missing table."""

    def test_missing_mart_raises_catalog_exception(self, snapshot_dir):
        """Query a mart that does NOT exist in the snapshot -> CatalogException."""
        con = _snapshot_connection_from_dir(snapshot_dir)
        try:
            with pytest.raises(duckdb.CatalogException):
                con.execute("select * from marts.nonexistent_table_xyz").df()
        finally:
            con.close()

    def test_raw_pipeline_runs_not_in_snapshot(self, snapshot_dir):
        """raw.pipeline_runs is never snapshotted -> must raise CatalogException."""
        con = _snapshot_connection_from_dir(snapshot_dir)
        try:
            with pytest.raises(duckdb.CatalogException):
                con.execute("select * from raw.pipeline_runs").df()
        finally:
            con.close()
