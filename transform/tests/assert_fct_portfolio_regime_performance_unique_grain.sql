-- Singular test: the grain of fct_portfolio_regime_performance must be one row per
-- (strategy, regime). Passes when this query returns zero rows.
select strategy, regime, count(*) as n
from {{ ref('fct_portfolio_regime_performance') }}
group by 1, 2
having count(*) > 1
