"""Cached, read-only access to the marts for the dashboard."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from mmi.portfolio import windows
from mmi.settings import settings
from mmi.utils.db import connect


def db_exists() -> bool:
    # Snapshot mode reads committed Parquet (no DB): "present" iff the snapshot dir holds a mart.
    # On MotherDuck the store is remote; locally we check the file exists.
    if settings.snapshot_mode:
        return settings.snapshot_dir.is_dir() and any(settings.snapshot_dir.glob("*.parquet"))
    return True if settings.use_motherduck else Path(settings.duckdb_path).exists()


def _snapshot_connection() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB whose schemas are the committed Parquet snapshot.

    Each ``<table>.parquet`` in ``snapshot_dir`` is registered as a view ``marts.<table>``, so the
    accessors' existing ``select ... from marts.<table>`` queries run unchanged — no live DB, no
    secrets. ``raw``/``marts`` are pre-created so a query for a mart with no Parquet file (e.g.
    ``raw.pipeline_runs``, never snapshotted) raises a clean missing-table CatalogException that
    ``query()`` swallows to an empty frame — exactly as a missing table does on a live DB.
    """
    con = duckdb.connect(":memory:")
    # Disable Python replacement scans: a missing table must raise a clean CatalogException (which
    # query() swallows to empty), NOT get silently "replaced" by a same-named Python object in the
    # call stack — e.g. data.ml_forecast (the accessor function) before that mart is snapshotted.
    con.execute("set python_enable_replacements=false")
    con.execute("create schema if not exists raw")
    con.execute("create schema if not exists marts")
    for path in sorted(settings.snapshot_dir.glob("*.parquet")):
        # The dir is ours, but escape defensively: '' for the path literal, "" for the identifier.
        safe_path = str(path).replace("'", "''")
        safe_name = path.stem.replace('"', '""')
        con.execute(
            f"create view marts.\"{safe_name}\" as select * from read_parquet('{safe_path}')"
        )
    return con


@st.cache_data(ttl=300, show_spinner=False)
def query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Run a read-only query against the live DB, or the Parquet snapshot in snapshot mode.

    A *missing table* (e.g. ML/AI marts before those steps run) returns an empty frame.
    Connection/auth errors and schema drift (a missing column) are NOT swallowed — they must
    surface rather than silently rendering every panel blank.
    """
    if not db_exists():
        return pd.DataFrame()
    con = _snapshot_connection() if settings.snapshot_mode else connect(read_only=True)
    try:
        return con.execute(sql, list(params) if params else []).df()
    except duckdb.CatalogException:
        return pd.DataFrame()
    finally:
        con.close()


def assets() -> pd.DataFrame:
    return query("select symbol, asset_class from marts.dim_asset order by asset_class, symbol")


def asset_daily(symbol: str) -> pd.DataFrame:
    return query(
        "select date, close, daily_return, vol_20d, ma_50 from marts.fct_asset_daily "
        "where symbol = ? order by date",
        (symbol,),
    )


def portfolio_windows() -> list[str]:
    """The backtest windows actually present in the marts, in canonical enum order.

    Drives the dashboard's window selector — only windows that have been computed are offered, so
    the picker shows one option until D6 lands the full three.
    """
    df = query("select distinct window_id from marts.fct_portfolio_returns")
    present = set(df["window_id"]) if not df.empty else set()
    return [w for w in windows.WINDOWS if w in present]


def portfolio_returns(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select strategy, date, daily_return, cumulative_return, drawdown, rolling_sharpe_252 "
        "from marts.fct_portfolio_returns where window_id = ? order by strategy, date",
        (window_id,),
    )


def portfolio_strategy_stats(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select strategy, sharpe, sharpe_lo, sharpe_hi, n_obs, n_boot, ci_pct "
        "from marts.fct_portfolio_strategy_stats where window_id = ? order by sharpe desc",
        (window_id,),
    )


def portfolio_strategy_pairs(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select strategy_a, strategy_b, sharpe_diff, diff_lo, diff_hi, distinguishable "
        "from marts.fct_portfolio_strategy_pairs where window_id = ? "
        "order by strategy_a, strategy_b",
        (window_id,),
    )


def portfolio_attribution(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select strategy, symbol, contribution_to_return, contribution_to_risk "
        "from marts.fct_performance_attribution where window_id = ? "
        "order by strategy, contribution_to_return",
        (window_id,),
    )


def portfolio_regime_performance(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select strategy, regime, n_days, day_share, ann_return, ann_vol, ann_sharpe "
        "from marts.fct_portfolio_regime_performance where window_id = ?",
        (window_id,),
    )


def portfolio_ml_gate(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select date, forecast_skill, forecast_weight "
        "from marts.fct_portfolio_ml_gate where window_id = ? order by date",
        (window_id,),
    )


def portfolio_btc_effect() -> pd.DataFrame:
    """The paired BTC effect per strategy (cross-window; no window filter). Empty until the two
    2015 windows have been computed."""
    return query(
        "select strategy, sharpe_ex, sharpe_inc, sharpe_diff, diff_lo, diff_hi, distinguishable "
        "from marts.fct_portfolio_btc_effect order by sharpe_diff"
    )


def crypto_intraday(symbol: str) -> pd.DataFrame:
    return query(
        "select ts, price_usd, pct_change from marts.fct_crypto_intraday "
        "where symbol = ? order by ts",
        (symbol,),
    )


def crypto_symbols() -> list[str]:
    df = query("select distinct symbol from marts.fct_crypto_intraday order by symbol")
    return df["symbol"].tolist() if not df.empty else []


def macro_ids() -> list[str]:
    df = query("select distinct series_id from marts.fct_macro_indicator order by series_id")
    return df["series_id"].tolist() if not df.empty else []


def macro(series_id: str) -> pd.DataFrame:
    return query(
        "select date, value, change from marts.fct_macro_indicator "
        "where series_id = ? order by date",
        (series_id,),
    )


def market_macro() -> pd.DataFrame:
    return query("select * from marts.fct_market_macro order by date")


def model_metrics() -> pd.DataFrame:
    return query("select model, symbol, metric, value, trained_at from marts.model_metrics")


def ml_forecast() -> pd.DataFrame:
    return query("select * from marts.ml_forecast")


def regimes(symbol: str) -> pd.DataFrame:
    return query(
        "select date, vol_20d, regime from marts.fct_regime where symbol = ? order by date",
        (symbol,),
    )


def latest_brief() -> pd.DataFrame:
    return query(
        "select created_at, engine, brief from marts.market_brief order by created_at desc limit 1"
    )


def pipeline_runs() -> pd.DataFrame:
    return query(
        "select source, rows, status, finished_at from raw.pipeline_runs "
        "order by started_at desc limit 12"
    )


def data_as_of() -> str:
    """The freshest markets date (max `date` in `fct_market_macro`), as an ISO string — `""` if no
    data. Drawn from a mart, so it is identical in live and snapshot mode (unlike `pipeline_runs`,
    whose `raw.pipeline_runs` is never snapshotted)."""
    # Cast to varchar in SQL so DuckDB yields a clean ISO date ("2026-06-23"); going through
    # pandas would coerce the DATE to a Timestamp and str() it as "... 00:00:00".
    df = query("select cast(max(date) as varchar) as d from marts.fct_market_macro")
    if df.empty or pd.isna(df["d"].iloc[0]):
        return ""
    return str(df["d"].iloc[0])


def is_sample_data() -> bool | None:
    """Provenance of the displayed markets data: `True` if ALL synthetic sample data (from
    `mmi seed`), `False` if ALL live (ingested), `None` if there's no data or the provenance is
    mixed/unrecorded. Read from `fct_asset_daily.source` (stamped `"sample"` by `mmi seed`, and
    with the source name by the extractors) so it stays honest in snapshot mode too, where
    `raw.pipeline_runs` is unavailable. A mixed set (e.g. a partial cron ingest leaving some
    symbols `"sample"` while others go `"yahoo"`) is deliberately `None`: never claim "live"
    unless every row is."""
    df = query("select distinct source from marts.fct_asset_daily")
    sources = set(df["source"].dropna()) if not df.empty else set()
    sources.discard("")  # a blank source carries no provenance signal
    if not sources:
        return None  # no rows, or only null/blank sources
    if sources == {"sample"}:
        return True  # purely synthetic
    if "sample" not in sources:
        return False  # purely ingested
    return None  # mixed sample + live → ambiguous; don't claim either
