-- Final daily asset fact: returns, rolling 20d volatility, 50d moving average.
with r as (
    select * from {{ ref('int_asset_returns') }}
)
select
    *,
    -- Annualised 20-day rolling volatility: daily std dev × √252.
    stddev_samp(daily_return) over (
        partition by symbol order by date rows between 20 preceding and current row
    ) * sqrt(252) as vol_20d,
    avg(close) over (
        partition by symbol order by date rows between 49 preceding and current row
    ) as ma_50
from r
