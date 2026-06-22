-- Singular test: the grain of fct_portfolio_returns must be one row per (strategy, date).
-- Passes when this query returns zero rows.
select strategy, date, count(*) as n
from {{ ref('fct_portfolio_returns') }}
group by 1, 2
having count(*) > 1
