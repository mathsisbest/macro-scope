-- Daily simple returns per asset (basis for vol / MA in the marts layer).
with base as (
    select * from {{ ref('stg_asset_prices') }}
)
select
    *,
    close / lag(close) over (partition by symbol order by date) - 1 as daily_return
from base
