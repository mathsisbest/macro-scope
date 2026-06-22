"""Raw source tables are pre-created (empty) so dbt always has its sources on a fresh DB."""

import pandas as pd

from mmi.ingestion.loader import _RAW_TABLES, DuckDBLoader, ensure_raw_tables

# Columns each dbt staging model selects from its raw source (must exist in the stub schema,
# else `dbt build` binder-errors on a fresh DB).
_STAGING_NEEDS = {
    "raw.asset_prices": {
        "symbol",
        "asset_class",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    },
    "raw.crypto_prices": {"symbol", "ts", "price_usd", "market_cap", "volume_24h", "source"},
    "raw.macro_series": {"series_id", "date", "value", "source"},
    "raw.worldbank": {"indicator_id", "country", "date", "value", "source"},
}


def test_ensure_raw_tables_creates_empty_sources_idempotently(con):
    ensure_raw_tables(con)
    ensure_raw_tables(con)  # idempotent — second call is a no-op
    for table in _RAW_TABLES:
        assert con.execute(f"select count(*) from {table}").fetchone()[0] == 0


def test_loader_init_creates_raw_sources(con):
    DuckDBLoader(con)
    for table in _RAW_TABLES:
        con.execute(f"select * from {table} limit 0")  # exists -> no CatalogException


def test_stub_has_every_column_staging_selects(con):
    ensure_raw_tables(con)
    for table, needed in _STAGING_NEEDS.items():
        cols = {row[0] for row in con.execute(f"describe {table}").fetchall()}
        assert needed <= cols, f"{table} missing {needed - cols}"


def test_upsert_into_precreated_table_matches_by_name(con):
    loader = DuckDBLoader(con)  # pre-creates raw.asset_prices with the canonical schema
    df = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "asset_class": ["equities"],
            "date": pd.to_datetime(["2026-06-20"], utc=True),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100.0],
            "source": ["stooq"],
        }
    )
    assert loader.upsert("raw.asset_prices", df, ["symbol", "date"]) == 1
    assert con.execute("select close from raw.asset_prices").fetchone()[0] == 1.5


# Columns each PORTFOLIO mart selects from its raw source (these are landed by `mmi portfolio`
# and read directly by the marts, not via a staging model).
_MART_NEEDS = {
    "raw.portfolio_returns": {"strategy", "date", "daily_return", "cumulative_return"},
    "raw.portfolio_strategy_stats": {
        "strategy",
        "sharpe",
        "sharpe_lo",
        "sharpe_hi",
        "n_obs",
        "n_boot",
        "ci_pct",
        "block_days",
    },
    "raw.portfolio_strategy_pairs": {
        "strategy_a",
        "strategy_b",
        "sharpe_a",
        "sharpe_b",
        "sharpe_diff",
        "diff_lo",
        "diff_hi",
        "distinguishable",
    },
}


def test_portfolio_stub_has_every_column_its_mart_selects(con):
    ensure_raw_tables(con)
    for table, needed in _MART_NEEDS.items():
        cols = {row[0] for row in con.execute(f"describe {table}").fetchall()}
        assert needed <= cols, f"{table} missing {needed - cols}"
