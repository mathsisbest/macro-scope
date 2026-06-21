-- Singular test: the grain of fct_asset_daily must be one row per (symbol, date).
-- Passes when this query returns zero rows.
select symbol, date, count(*) as n
from {{ ref('fct_asset_daily') }}
group by 1, 2
having count(*) > 1
