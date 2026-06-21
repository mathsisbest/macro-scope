-- Markets in macro context: SPY joined to the latest-available yields via ASOF JOIN
-- (DuckDB native), so business-daily prices line up with lower-frequency macro data.
with spy as (
    select date, close, daily_return, vol_20d
    from {{ ref('fct_asset_daily') }}
    where symbol = 'SPY'
),
y10 as (
    select date, value from {{ ref('fct_macro_indicator') }} where series_id = 'DGS10'
),
y2 as (
    select date, value from {{ ref('fct_macro_indicator') }} where series_id = 'DGS2'
)
select
    spy.date,
    spy.close        as spy_close,
    spy.daily_return as spy_return,
    spy.vol_20d,
    y10.value        as us_10y,
    y2.value         as us_2y,
    y10.value - y2.value as yield_curve_10y_2y
from spy
asof left join y10 on spy.date >= y10.date
asof left join y2  on spy.date >= y2.date
