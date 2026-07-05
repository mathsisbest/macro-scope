with source as (
    select * from {{ source('raw', 'macro_series') }}
)
select
    series_id,
    cast(date as date)   as date,
    cast(value as double) as value,
    source,
    loaded_at
from source
where value is not null
