select
    symbol,
    ts,
    price_usd,
    market_cap,
    volume_24h,
    price_usd / lag(price_usd) over (partition by symbol order by ts) - 1 as pct_change
from {{ ref('stg_crypto_prices') }}
