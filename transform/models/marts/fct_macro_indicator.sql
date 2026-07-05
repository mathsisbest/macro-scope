select
    series_id,
    date,
    value,
    value - lag(value) over (partition by series_id order by date) as change,
    loaded_at
from {{ ref('stg_macro_series') }}
