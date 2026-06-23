with source as (
    select * from {{ source('raw', 'asset_prices') }}
    where close is not null
),
deduped as (
    -- Yahoo's v8 chart returns a live/partial bar for the *current* day on top of the day's
    -- regular bar (seen for 24h FX like EURUSD/GBPUSD); both collapse to the same calendar date
    -- once truncated, breaking the one-row-per-(symbol, date) grain. Keep the latest quote per day.
    select *
    from source
    qualify row_number() over (
        partition by symbol, cast(date as date) order by date desc
    ) = 1
)
select
    symbol,
    asset_class,
    cast(date as date)    as date,
    cast(open as double)  as open,
    cast(high as double)  as high,
    cast(low as double)   as low,
    cast(close as double) as close,
    cast(volume as bigint) as volume,
    source
from deduped
