-- Per-strategy daily portfolio returns from raw.portfolio_returns (computed by `mmi portfolio`),
-- with SQL-derived drawdown and a rolling 252-day annualised Sharpe. Tested grain:
-- (window_id, strategy, date). The rolling windows partition by (window_id, strategy) so a peak or
-- a 252-day Sharpe window never bleeds across windows.
-- Run order: `mmi portfolio` must land raw.portfolio_returns before this model builds.
with source as (
    select
        window_id,
        strategy,
        cast(date as date) as date,
        daily_return,
        cumulative_return,
        1 + cumulative_return as wealth
    from {{ source('raw', 'portfolio_returns') }}
)
select
    window_id,
    strategy,
    date,
    daily_return,
    cumulative_return,
    wealth / max(wealth) over (
        partition by window_id, strategy order by date rows between unbounded preceding and current row
    ) - 1 as drawdown,
    avg(daily_return) over w
        / nullif(stddev_samp(daily_return) over w, 0)
        * sqrt(252) as rolling_sharpe_252
from source
window w as (
    partition by window_id, strategy order by date rows between 251 preceding and current row
)
