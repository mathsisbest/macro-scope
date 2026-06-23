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
    # On MotherDuck the store is remote; locally we check the file exists.
    return True if settings.use_motherduck else Path(settings.duckdb_path).exists()


@st.cache_data(ttl=300, show_spinner=False)
def query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Run a read-only query.

    A *missing table* (e.g. ML/AI marts before those steps run) returns an empty frame.
    Connection/auth errors and schema drift (a missing column) are NOT swallowed — they must
    surface rather than silently rendering every panel blank.
    """
    if not db_exists():
        return pd.DataFrame()
    con = connect(read_only=True)
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
