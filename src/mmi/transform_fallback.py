"""Pure-SQL fallback that builds the marts when dbt isn't installed.

dbt (in ``transform/``) is the *canonical* transform layer. This module mirrors the
same logic so ``make demo`` works out-of-the-box without the dbt dependency. The two
are intentionally kept simple and equivalent; dbt adds tests, docs and lineage on top.
"""

from __future__ import annotations

from mmi.utils.db import init_schemas
from mmi.utils.logging import get_logger

log = get_logger("transform_fallback")

_SQL = [
    # dim_asset -------------------------------------------------------------
    """
    CREATE OR REPLACE TABLE marts.dim_asset AS
    SELECT DISTINCT symbol, asset_class FROM raw.asset_prices;
    """,
    # fct_asset_daily: returns + rolling vol + moving average -----------------
    """
    CREATE OR REPLACE TABLE marts.fct_asset_daily AS
    WITH base AS (
        SELECT symbol, asset_class, CAST(date AS DATE) AS date, open, high, low, close, volume
        FROM raw.asset_prices
    ), ret AS (
        SELECT *,
            close / LAG(close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS daily_return
        FROM base
    )
    SELECT *,
        STDDEV_SAMP(daily_return) OVER (
            PARTITION BY symbol ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ) * SQRT(252.0) AS vol_20d,
        AVG(close) OVER (
            PARTITION BY symbol ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ) AS ma_50
    FROM ret;
    """,
    # fct_macro_indicator -----------------------------------------------------
    """
    CREATE OR REPLACE TABLE marts.fct_macro_indicator AS
    SELECT series_id, CAST(date AS DATE) AS date, value,
        value - LAG(value) OVER (PARTITION BY series_id ORDER BY date) AS change
    FROM raw.macro_series;
    """,
    # fct_market_macro: markets in macro context via ASOF join ----------------
    """
    CREATE OR REPLACE TABLE marts.fct_market_macro AS
    WITH spy AS (
        SELECT date, close, daily_return, vol_20d FROM marts.fct_asset_daily WHERE symbol = 'SPY'
    ), y10 AS (
        SELECT date, value FROM marts.fct_macro_indicator WHERE series_id = 'DGS10'
    ), y2 AS (
        SELECT date, value FROM marts.fct_macro_indicator WHERE series_id = 'DGS2'
    )
    SELECT spy.date, spy.close AS spy_close, spy.daily_return AS spy_return, spy.vol_20d,
        y10.value AS us_10y, y2.value AS us_2y, (y10.value - y2.value) AS yield_curve_10y_2y
    FROM spy
    ASOF LEFT JOIN y10 ON spy.date >= y10.date
    ASOF LEFT JOIN y2  ON spy.date >= y2.date;
    """,
    # fct_portfolio_returns ---------------------------------------------------
    """
    CREATE OR REPLACE TABLE marts.fct_portfolio_returns AS
    WITH source AS (
        SELECT
            window_id,
            strategy,
            CAST(date AS DATE) AS date,
            daily_return,
            cumulative_return,
            1 + cumulative_return AS wealth
        FROM raw.portfolio_returns
    )
    SELECT
        window_id,
        strategy,
        date,
        daily_return,
        cumulative_return,
        wealth / MAX(wealth) OVER (
            PARTITION BY window_id, strategy
            ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) - 1 AS drawdown,
        AVG(daily_return) OVER w
            / NULLIF(STDDEV_SAMP(daily_return) OVER w, 0)
            * SQRT(252) AS rolling_sharpe_252
    FROM source
    WINDOW w AS (
        PARTITION BY window_id, strategy ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
    );
    """,
    # fct_portfolio_strategy_stats -------------------------------------------
    """
    CREATE OR REPLACE TABLE marts.fct_portfolio_strategy_stats AS
    SELECT * FROM raw.portfolio_strategy_stats;
    """,
    # fct_portfolio_strategy_pairs -------------------------------------------
    """
    CREATE OR REPLACE TABLE marts.fct_portfolio_strategy_pairs AS
    SELECT * FROM raw.portfolio_strategy_pairs;
    """,
    # fct_portfolio_attribution ----------------------------------------------
    """
    CREATE OR REPLACE TABLE marts.fct_portfolio_attribution AS
    SELECT * FROM raw.portfolio_attribution;
    """,
    # fct_portfolio_btc_effect -----------------------------------------------
    """
    CREATE OR REPLACE TABLE marts.fct_portfolio_btc_effect AS
    SELECT * FROM raw.portfolio_btc_effect;
    """,
    # fct_portfolio_ml_gate --------------------------------------------------
    """
    CREATE OR REPLACE TABLE marts.fct_portfolio_ml_gate AS
    SELECT * FROM raw.portfolio_ml_gate;
    """,
]


def build_marts(con) -> None:
    """Create all marts tables from raw via SQL."""
    init_schemas(con)
    for stmt in _SQL:
        con.execute(stmt)
    log.info("built %d marts tables (fallback)", len(_SQL))
