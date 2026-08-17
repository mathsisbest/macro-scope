"""Covers _load_portfolio_macro_context: the DuckDB PIVOT + forward-fill pushdown.

The macro wide-frame must match the old pandas pivot_table + ffill semantics exactly
(audit item R4), including leading NaNs before a series starts, carried values at
reading-absent dates, and non-null-wins tiebreaks for duplicate (date, series) rows.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from mmi.cli import _load_portfolio_macro_context


def _db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("create schema marts")
    con.execute(
        "create table marts.fct_macro_indicator (date date, series_id varchar, value double)"
    )
    con.execute(
        "create table marts.fct_asset_daily (symbol varchar, date date, daily_return double)"
    )
    return con


def _seed(con, rows) -> None:
    con.execute("delete from marts.fct_macro_indicator")
    con.executemany(
        "insert into marts.fct_macro_indicator values (?, ?, ?)",
        [(pd.Timestamp(d).date(), s, v) for d, s, v in rows],
    )


def test_macro_context_wide_shape_and_carry_forward():
    con = _db()
    # CPI missing 2020-01-02 (must be carried); UNRATE missing 2020-01-01 only.
    _seed(
        con,
        [
            ("2020-01-01", "CPI", 10.0),
            ("2020-01-02", "UNRATE", 5.1),
            ("2020-01-03", "CPI", 11.0),
            ("2020-01-03", "UNRATE", 4.5),
        ],
    )
    macro_wide, _ = _load_portfolio_macro_context(con)
    assert macro_wide is not None
    assert list(macro_wide.columns) == ["date", "CPI", "UNRATE"]
    assert macro_wide["date"].dtype == "datetime64[ns]"
    assert list(macro_wide["date"]) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    row_01 = macro_wide[macro_wide["date"] == pd.Timestamp("2020-01-01")].iloc[0]
    # UNRATE leads with NaN until its first reading (pandas ffill leaves leading NaNs).
    assert pd.isna(row_01["UNRATE"])
    row_02 = macro_wide[macro_wide["date"] == pd.Timestamp("2020-01-02")].iloc[0]
    # CPI carried forward one day, exactly like pandas ffill.
    assert row_02["CPI"] == 10.0
    assert row_02["UNRATE"] == 5.1
    row_03 = macro_wide[macro_wide["date"] == pd.Timestamp("2020-01-03")].iloc[0]
    assert row_03["CPI"] == 11.0
    assert row_03["UNRATE"] == 4.5


def test_macro_context_leading_nan_before_series_starts():
    con = _db()
    # UNRATE first reading is 2020-01-02, one day after the grid starts.
    _seed(
        con,
        [
            ("2020-01-01", "CPI", 10.0),
            ("2020-01-02", "CPI", 10.1),
            ("2020-01-02", "UNRATE", 4.9),
        ],
    )
    macro_wide, _ = _load_portfolio_macro_context(con)
    # 2020-01-01 stays NaN for UNRATE (pandas ffill leaves leading NaNs).
    row_01 = macro_wide[macro_wide["date"] == pd.Timestamp("2020-01-01")].iloc[0]
    assert pd.isna(row_01["UNRATE"])
    row_02 = macro_wide[macro_wide["date"] == pd.Timestamp("2020-01-02")].iloc[0]
    assert row_02["UNRATE"] == 4.9


def test_macro_context_duplicate_rows_non_null_wins():
    con = _db()
    _seed(
        con,
        [
            ("2020-01-01", "CPI", None),
            ("2020-01-01", "CPI", 10.0),
            ("2020-01-01", "UNRATE", 5.0),
            ("2020-01-02", "CPI", None),
            ("2020-01-02", "UNRATE", 5.0),
        ],
    )
    macro_wide, _ = _load_portfolio_macro_context(con)
    row_01 = macro_wide[macro_wide["date"] == "2020-01-01"].iloc[0]
    assert row_01["CPI"] == 10.0
    row_02 = macro_wide[macro_wide["date"] == "2020-01-02"].iloc[0]
    # 2020-01-02 has no CPI reading: carried from 2020-01-01.
    assert row_02["CPI"] == 10.0


def test_macro_context_empty_and_asset_dfs():
    con = _db()
    macro_wide, asset_dfs = _load_portfolio_macro_context(con)
    assert macro_wide is None
    assert asset_dfs == {}
