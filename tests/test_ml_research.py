"""Tests for mmi.ml.research and mmi.ml.research_forecast."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from mmi.ml.research import (
    _load_asset_data,
    _load_macro_data,
    _model_params,
    run_research,
)
from mmi.ml.research_forecast import (
    _load_asset_vol,
    _pivot_macro,
    _spy_df,
    run_sweep,
    summarize,
)


@pytest.fixture
def mock_con():
    """Create an in-memory DuckDB database seeded with minimal schema and data."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA marts")
    con.execute("""
        CREATE TABLE marts.fct_asset_daily (
            symbol VARCHAR,
            date TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            daily_return DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE marts.fct_macro_indicator (
            date TIMESTAMP,
            series_id VARCHAR,
            value DOUBLE
        )
    """)

    # Seed SPY daily data (100 rows)
    dates = pd.bdate_range("2023-01-01", periods=100)
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, size=100)

    spy_rows = []
    for d, r in zip(dates, rets):
        spy_rows.append(("SPY", d, 100.0, 101.0, 99.0, 100.0 + r, r))

    con.executemany(
        "INSERT INTO marts.fct_asset_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
        spy_rows,
    )

    # Seed macro data
    macro_rows = []
    for d in dates:
        macro_rows.append((d, "VIXCLS", 18.5))
        macro_rows.append((d, "T10Y2Y", 0.25))

    con.executemany(
        "INSERT INTO marts.fct_macro_indicator VALUES (?, ?, ?)",
        macro_rows,
    )

    yield con
    con.close()


def test_model_params():
    assert _model_params("rv_har") == [{}]
    assert len(_model_params("rv_ridge")) == 3
    assert len(_model_params("rv_lasso")) == 3
    assert len(_model_params("rv_gb")) == 3
    assert _model_params("unknown_model") == [{}]


def test_load_macro_data_empty():
    empty_con = duckdb.connect(":memory:")
    df = _load_macro_data(empty_con)
    assert df.empty
    empty_con.close()


def test_load_macro_data_with_data(mock_con):
    df = _load_macro_data(mock_con)
    assert not df.empty
    assert "date" in df.columns
    assert "VIXCLS" in df.columns


def test_load_asset_data(mock_con):
    asset_dfs = _load_asset_data(mock_con, ["SPY", "NONEXISTENT"])
    assert "SPY" in asset_dfs
    assert "NONEXISTENT" not in asset_dfs
    assert len(asset_dfs["SPY"]) == 100


def test_run_research_smoke(mock_con):
    results = run_research(
        mock_con,
        symbol="SPY",
        models=["rv_har"],
        feature_sets=["vol"],
        horizons=[5],
        n_splits_list=[5],
    )
    assert isinstance(results, pd.DataFrame)


def test_research_forecast_helpers(mock_con):
    # Pivot macro
    pivoted = _pivot_macro(mock_con)
    assert not pivoted.empty
    assert "VIXCLS" in pivoted.columns
    assert "T10Y2Y" in pivoted.columns

    # None con
    assert _pivot_macro(None).empty

    # Asset vol
    asset_vol = _load_asset_vol(mock_con, ("SPY",))
    assert "SPY" in asset_vol
    assert len(asset_vol["SPY"]) == 100

    # SPY df
    spy = _spy_df(mock_con)
    assert len(spy) == 100
    assert "daily_return" in spy.columns


def test_run_sweep_missing_symbol_raises():
    empty_con = duckdb.connect(":memory:")
    empty_con.execute("CREATE SCHEMA marts")
    empty_con.execute("""
        CREATE TABLE marts.fct_asset_daily (
            symbol VARCHAR, date TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, daily_return DOUBLE
        )
    """)
    empty_con.execute("CREATE TABLE marts.fct_macro_indicator (date TIMESTAMP, series_id VARCHAR, value DOUBLE)")
    with pytest.raises(ValueError, match="No data for UNKNOWN"):
        run_sweep(empty_con, symbol="UNKNOWN")
    empty_con.close()


def test_summarize():
    df = pd.DataFrame({
        "model": ["gb", "lgb"],
        "feature_set": ["default", "vol_medium"],
        "target_type": ["raw", "raw"],
        "horizon": [1, 5],
        "ic": [0.05, 0.12],
        "direction_accuracy": [0.52, 0.55],
        "prediction_count": [100, 100],
        "sharpe": [0.8, 1.2],
    })
    top = summarize(df, top_k=1)
    assert len(top) == 1
    assert top.iloc[0]["model"] == "lgb"
