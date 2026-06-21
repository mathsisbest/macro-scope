select distinct symbol, asset_class from {{ ref('stg_asset_prices') }}
union
select distinct symbol, 'crypto' as asset_class from {{ ref('stg_crypto_prices') }}
