with source as (
    select * from {{ source('raw', 'worldbank') }}
)
select
    indicator_id,
    country,
    cast(date as integer) as year,
    cast(value as double)  as value,
    source
from source
where value is not null
