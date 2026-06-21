with source as (
    select * from {{ source('raw', 'crypto_prices') }}
)
select
    symbol,
    cast(ts as timestamp)        as ts,
    cast(price_usd as double)    as price_usd,
    cast(market_cap as double)   as market_cap,
    cast(volume_24h as double)   as volume_24h,
    source
from source
where price_usd is not null
