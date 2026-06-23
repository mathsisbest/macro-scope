-- Contract: the two 2015 windows (ex_btc_2015, inc_btc_2015) must be PERIOD-IDENTICAL — the exact
-- same (strategy, date) set — so the paired BTC-effect bootstrap (which co-resamples the same dates
-- in both) is valid and the BTC effect is a clean same-period comparison. A FULL OUTER JOIN surfaces
-- any date present in one window but not the other (endpoints alone would not catch an interior
-- gap). Passes when this returns zero rows. (Vacuously true on a partial DB where the 2015 windows
-- have not been computed.)
with ex as (
    select strategy, date from {{ ref('fct_portfolio_returns') }} where window_id = 'ex_btc_2015'
),

inc as (
    select strategy, date from {{ ref('fct_portfolio_returns') }} where window_id = 'inc_btc_2015'
)

select
    coalesce(ex.strategy, inc.strategy) as strategy,
    coalesce(ex.date, inc.date) as date,
    case when ex.date is null then 'missing in ex_btc_2015' else 'missing in inc_btc_2015' end as issue
from ex
full outer join inc on ex.strategy = inc.strategy and ex.date = inc.date
where ex.date is null or inc.date is null
