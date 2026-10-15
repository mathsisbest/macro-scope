"""Tests for R3: Equity Risk Premium (ERP) chain, real rates (DFII10), and valuation charts."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest
from dashboard import data
from dashboard.components import charts


@pytest.fixture
def sample_valuation_db(tmp_path):
    """Create a temporary DuckDB database with mock macro data for DGS10 and DFII10."""
    db_path = tmp_path / "test_val.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("create schema if not exists marts")
    con.execute(
        """
        create table marts.fct_macro_indicator as
        select * from (values
            ('DGS10', DATE '2024-01-01', 4.0, 0.0, TIMESTAMP '2024-01-01 00:00:00'),
            ('DGS10', DATE '2024-01-02', 4.2, 0.2, TIMESTAMP '2024-01-02 00:00:00'),
            ('DFII10', DATE '2024-01-01', 1.8, 0.0, TIMESTAMP '2024-01-01 00:00:00'),
            ('DFII10', DATE '2024-01-02', 1.9, 0.1, TIMESTAMP '2024-01-02 00:00:00')
        ) as t(series_id, date, value, change, loaded_at)
        """
    )
    con.close()
    return db_path


def test_dfii10_in_assets_config():
    """Verify DFII10 is configured in config/assets.yml."""
    from mmi.settings import load_assets

    assets = load_assets()
    macro_series = assets.get("macro", [])
    ids = [s["id"] for s in macro_series]
    assert "DFII10" in ids

    item = next(s for s in macro_series if s["id"] == "DFII10")
    assert item["category"] == "Rates & curve"
    assert item["units"] == "%"


def test_dfii10_in_sampledata():
    """Verify sampledata generates DFII10 with reasonable defaults."""
    from mmi.sampledata import _macro

    df = _macro()
    assert not df.empty
    df_tips = df[df["series_id"] == "DFII10"]
    assert not df_tips.empty
    assert (df_tips["value"] > 0).all()


def test_valuation_data_empty_when_no_dgs10(monkeypatch, tmp_path):
    """When DGS10 data is absent, valuation_data returns an empty DataFrame with proper columns."""
    empty_db = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(empty_db))
    con.execute("create schema if not exists marts")
    con.execute(
        "create table marts.fct_macro_indicator "
        "(series_id varchar, date date, value double, change double, loaded_at timestamp)"
    )
    con.close()

    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(
        data, "connect", lambda *a, **k: duckdb.connect(str(empty_db), read_only=True)
    )
    data.query.clear()

    df = data.valuation_data()
    assert list(df.columns) == ["date", "us_10y", "earn_yield", "erp", "cape", "tips_10y"]
    assert df.empty


def test_valuation_data_computes_erp(monkeypatch, sample_valuation_db, tmp_path):
    """Verify valuation_data properly computes ERP = earn_yield - us_10y."""
    # Write mock Shiller CAPE parquet
    shiller_df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "cape": [20.0],
            "earn_yield": [0.05],  # 5%
        }
    )
    shiller_path = tmp_path / "shiller_cape.parquet"
    shiller_df.to_parquet(shiller_path)

    monkeypatch.setattr(data.settings, "snapshot_dir", tmp_path)
    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(
        data, "connect", lambda *a, **k: duckdb.connect(str(sample_valuation_db), read_only=True)
    )
    data.query.clear()

    df = data.valuation_data()
    assert not df.empty
    assert "erp" in df.columns
    # earn_yield is 5.0 (0.05 * 100), us_10y is 4.0 -> ERP = 1.0
    row0 = df.iloc[0]
    assert row0["us_10y"] == 4.0
    assert row0["earn_yield"] == 5.0
    assert row0["erp"] == 1.0
    assert row0["tips_10y"] == 1.8


def test_erp_chart_renders_all_traces():
    """Verify erp_chart produces traces for ERP, earnings yield, nominal yield, and real rates."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "us_10y": [4.0, 4.2],
            "earn_yield": [5.0, 5.0],
            "erp": [1.0, 0.8],
            "cape": [20.0, 20.0],
            "tips_10y": [1.8, 1.9],
        }
    )
    fig = charts.erp_chart(df)
    trace_names = [t.name for t in fig.data]
    assert "Equity Risk Premium (ERP)" in trace_names
    assert "S&P 500 Earnings Yield (1/CAPE)" in trace_names
    assert "10Y Treasury Yield (DGS10)" in trace_names
    assert "10Y TIPS Real Yield (DFII10)" in trace_names

    # Check zero threshold line
    shapes = fig.layout.shapes
    assert any(s.y0 == 0 and s.y1 == 0 for s in shapes)


def test_cape_ratio_chart_renders():
    """Verify cape_ratio_chart produces CAPE trace and median reference line."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "cape": [25.0, 25.5],
        }
    )
    fig = charts.cape_ratio_chart(df)
    trace_names = [t.name for t in fig.data]
    assert "Shiller CAPE" in trace_names
    shapes = fig.layout.shapes
    assert any(s.y0 == 16.8 and s.y1 == 16.8 for s in shapes)
