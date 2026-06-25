with source as (
    select * from {{ source('raw', 'asset_prices') }}
    where close is not null
      -- Yahoo injects a PHANTOM volume-0 bar (midnight-UTC, ~half the real price) for some ETFs —
      -- notably on market holidays, where it is the ONLY bar for that date, so the dedup below keeps
      -- it and the bad price corrupts daily returns (±50%+ spikes into/out of the holiday). Drop
      -- volume-less bars for assets that trade with volume; FX legitimately has no volume, so it is
      -- exempt. Verified: removes 26 out-of-bounds returns while preserving every real trading date.
      and not (coalesce(volume, 0) = 0 and asset_class <> 'fx')
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
