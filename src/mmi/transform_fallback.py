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
        SELECT symbol, asset_class, CAST(date AS DATE) AS date, open, high, low, close, volume,
            COALESCE(source, 'yahoo') AS source
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
        value - LAG(value) OVER (PARTITION BY series_id ORDER BY date) AS change,
        COALESCE(source, 'fred') AS source,
        loaded_at
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
    ), y3mo AS (
        SELECT date, value FROM marts.fct_macro_indicator WHERE series_id = 'DGS3MO'
    )
    SELECT spy.date, spy.close AS spy_close, spy.daily_return AS spy_return, spy.vol_20d,
        y10.value AS us_10y, y2.value AS us_2y, y3mo.value AS us_3m,
        (y10.value - y2.value) AS yield_curve_10y_2y,
        (y10.value - y3mo.value) AS yield_curve_10y_3m
    FROM spy
    ASOF LEFT JOIN y10  ON spy.date >= y10.date
    ASOF LEFT JOIN y2   ON spy.date >= y2.date
    ASOF LEFT JOIN y3mo ON spy.date >= y3mo.date;
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
    # fct_portfolio_regime_performance ---------------------------------------
    """
    CREATE OR REPLACE TABLE marts.fct_portfolio_regime_performance AS
    WITH window_bounds AS (
        SELECT window_id, MIN(date) AS lo, MAX(date) AS hi
        FROM marts.fct_portfolio_returns
        WHERE daily_return <> 0
        GROUP BY window_id
    ),
    first_invested AS (
        SELECT window_id, strategy, MIN(date) AS start_date
        FROM marts.fct_portfolio_returns
        WHERE daily_return <> 0
        GROUP BY window_id, strategy
    ),
    spy AS (
        SELECT date, vol_20d
        FROM marts.fct_asset_daily
        WHERE symbol = 'SPY' AND vol_20d IS NOT NULL
    ),
    spy_regime AS (
        SELECT
            b.window_id,
            s.date,
            CASE NTILE(3) OVER (PARTITION BY b.window_id ORDER BY s.vol_20d)
                WHEN 1 THEN 'Low'
                WHEN 2 THEN 'Medium'
                ELSE 'High'
            END AS regime
        FROM window_bounds AS b
        JOIN spy AS s ON s.date BETWEEN b.lo AND b.hi
    ),
    joined AS (
        SELECT p.window_id, p.strategy, r.regime, p.daily_return
        FROM marts.fct_portfolio_returns AS p
        JOIN first_invested AS fi
            ON fi.window_id = p.window_id AND fi.strategy = p.strategy AND p.date >= fi.start_date
        JOIN spy_regime AS r ON r.window_id = p.window_id AND r.date = p.date
    )
    SELECT
        window_id,
        strategy,
        regime,
        COUNT(*) AS n_days,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY window_id, strategy) AS day_share,
        AVG(daily_return) * 252 AS ann_return,
        STDDEV_SAMP(daily_return) * SQRT(252) AS ann_vol,
        CASE
            WHEN STDDEV_SAMP(daily_return) = 0 THEN NULL
            ELSE AVG(daily_return) / STDDEV_SAMP(daily_return) * SQRT(252)
        END AS ann_sharpe
    FROM joined
    GROUP BY window_id, strategy, regime;
    """,
    # fct_recession_risk ----------------------------------------------------
    """
    CREATE OR REPLACE TABLE marts.fct_recession_risk AS
    WITH y10 AS (
        SELECT date, value AS dgs10
        FROM marts.fct_macro_indicator
        WHERE series_id = 'DGS10'
    ),
    y3mo AS (
        SELECT date, value AS dgs3mo
        FROM marts.fct_macro_indicator
        WHERE series_id = 'DGS3MO'
    ),
    y2 AS (
        SELECT date, value AS dgs2
        FROM marts.fct_macro_indicator
        WHERE series_id = 'DGS2'
    ),
    joined AS (
        SELECT
            y10.date,
            y10.dgs10,
            y3mo.dgs3mo,
            y2.dgs2
        FROM y10
        ASOF LEFT JOIN y3mo ON y10.date >= y3mo.date
        ASOF LEFT JOIN y2   ON y10.date >= y2.date
    ),
    with_spread AS (
        SELECT
            date,
            CASE
                WHEN dgs3mo IS NOT NULL THEN dgs10 - dgs3mo
                ELSE                         dgs10 - dgs2
            END AS spread_10y_3m,
            CASE
                WHEN dgs3mo IS NOT NULL THEN '10y_3m'
                ELSE                         '10y_2y_proxy'
            END AS model
        FROM joined
        WHERE dgs10 IS NOT NULL
          AND (dgs3mo IS NOT NULL OR dgs2 IS NOT NULL)
    ),
    with_index AS (
        SELECT
            date,
            spread_10y_3m,
            model,
            -0.5333 + (-0.6629 * spread_10y_3m) AS z
        FROM with_spread
    ),
    with_prob AS (
        SELECT
            date,
            spread_10y_3m,
            model,
            z,
            1.0 / (1.0 + 0.2316419 * ABS(z))                  AS t,
            EXP(-0.5 * z * z) / SQRT(2.0 * 3.141592653589793) AS pdf,
            t * (0.319381530
                + t * (-0.356563782
                    + t * (1.781477937
                        + t * (-1.821255978
                            + t * 1.330274429))))             AS cdf_approx
        FROM with_index
    )
    SELECT
        date,
        spread_10y_3m,
        CASE WHEN z >= 0 THEN 1.0 - pdf * cdf_approx
             ELSE                 pdf * cdf_approx
        END AS recession_prob,
        model
    FROM with_prob
    ORDER BY date;
    """,
]


def build_marts(con) -> None:
    """Create all marts tables from raw via SQL."""
    init_schemas(con)
    for stmt in _SQL:
        con.execute(stmt)
    log.info("built %d marts tables (fallback)", len(_SQL))
