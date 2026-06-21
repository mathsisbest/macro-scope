with source as (
    select * from {{ source('raw', 'asset_prices') }}
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
from source
where close is not null
