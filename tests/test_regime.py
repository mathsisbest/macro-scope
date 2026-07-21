"""Tests for mmi.ml.regime — volatility-regime labelling."""

from __future__ import annotations

import duckdb
import pandas as pd

from mmi.ml.regime import label_regimes


def _con_with_data(sql: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("create schema marts")
    con.execute(sql)
    return con


def _has_expected_columns(df: pd.DataFrame) -> bool:
    return set(df.columns) == {"symbol", "date", "vol_20d", "regime"}


class TestLabelRegimes:
    def test_three_symbols_produces_all_labels(self):
        con = _con_with_data(
            "create table marts.fct_asset_daily as "
            "select 'SPY' as symbol, '2024-01-01'::date + i::int as date, "
            "  0.1 + 0.01 * (i % 10) as vol_20d "
            "from generate_series(0, 29) t(i) "
            "union all "
            "select 'TLT', '2024-01-01'::date + i::int, "
            "  0.2 + 0.01 * (i % 10) "
            "from generate_series(0, 29) t(i) "
            "union all "
            "select 'GLD', '2024-01-01'::date + i::int, "
            "  0.15 + 0.01 * (i % 10) "
            "from generate_series(0, 29) t(i)"
        )
        result = label_regimes(con)
        con.close()

        assert _has_expected_columns(result)
        assert len(result) == 90
        for sym in ["SPY", "TLT", "GLD"]:
            sym_labels = result.loc[result["symbol"] == sym, "regime"]
            assert sym_labels.isin(["Low", "Medium", "High"]).all()
            assert sym_labels.nunique() == 3

    def test_empty_table_returns_empty_frame(self):
        con = _con_with_data(
            "create table marts.fct_asset_daily as "
            "select * from (values "
            "(null::varchar, null::date, null::double)"
            ") t(symbol, date, vol_20d) "
            "where 1=0"
        )
        result = label_regimes(con)
        con.close()
        assert _has_expected_columns(result)
        assert result.empty

    def test_all_identical_vol_values(self):
        con = _con_with_data(
            "create table marts.fct_asset_daily as "
            "select 'SPY' as symbol, '2024-01-01'::date + i::int as date, "
            "  0.1 as vol_20d "
            "from generate_series(0, 8) t(i)"
        )
        result = label_regimes(con)
        con.close()
        assert _has_expected_columns(result)
        assert len(result) == 9
        assert result["regime"].nunique() == 3

    def test_single_symbol(self):
        con = _con_with_data(
            "create table marts.fct_asset_daily as "
            "select 'SPY' as symbol, '2024-01-01'::date + i::int as date, "
            "  (i + 1)::double / 100.0 as vol_20d "
            "from generate_series(0, 11) t(i)"
        )
        result = label_regimes(con)
        con.close()
        assert _has_expected_columns(result)
        assert len(result) == 12
        assert result["regime"].nunique() == 3

    def test_nan_vol_excluded(self):
        con = _con_with_data(
            "create table marts.fct_asset_daily as "
            "select 'SPY' as symbol, '2024-01-01'::date + i::int as date, "
            "  case when i % 4 = 2 then null else 0.1 + 0.01 * i end as vol_20d "
            "from generate_series(0, 11) t(i)"
        )
        result = label_regimes(con)
        con.close()
        assert _has_expected_columns(result)
        assert len(result) == 9  # 12 - 3 nulls
        assert result["regime"].nunique() == 3

    def test_regime_labels(self):
        con = _con_with_data(
            "create table marts.fct_asset_daily as "
            "select 'SPY' as symbol, '2024-01-01'::date + i::int as date, "
            "  (i + 1)::double / 100.0 as vol_20d "
            "from generate_series(0, 8) t(i)"
        )
        result = label_regimes(con)
        con.close()
        assert set(result["regime"].unique()) == {"Low", "Medium", "High"}

    def test_two_symbols_independent_terciles(self):
        con = _con_with_data(
            "create table marts.fct_asset_daily as "
            "select 'SPY' as symbol, '2024-01-01'::date + i::int as date, "
            "  (i + 1)::double / 100.0 as vol_20d "
            "from generate_series(0, 8) t(i) "
            "union all "
            "select 'TLT', '2024-01-02'::date + i::int, "
            "  (i + 10)::double / 100.0 "
            "from generate_series(0, 8) t(i)"
        )
        result = label_regimes(con)
        con.close()
        assert len(result) == 18
        for sym in ["SPY", "TLT"]:
            assert result.loc[result["symbol"] == sym, "regime"].nunique() == 3

    def test_fewer_than_3_rows_per_symbol(self):
        con = _con_with_data(
            "create table marts.fct_asset_daily as "
            "select 'SPY' as symbol, '2024-01-01'::date + i::int as date, "
            "  (i + 1)::double / 100.0 as vol_20d "
            "from generate_series(0, 1) t(i)"
        )
        result = label_regimes(con)
        con.close()
        assert _has_expected_columns(result)
        assert result.empty
