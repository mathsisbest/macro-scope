-- One row per tracked asset. Sourced solely from stg_asset_prices (Yahoo daily), which already
-- includes BTC under asset_class 'crypto' via the crypto_daily path.
select distinct symbol, asset_class from {{ ref('stg_asset_prices') }}
