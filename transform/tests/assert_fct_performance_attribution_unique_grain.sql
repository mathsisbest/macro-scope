-- Singular test: the grain of fct_performance_attribution must be one row per (strategy, symbol),
-- including the '(costs)' pseudo-symbol. Passes when this query returns zero rows.
select strategy, symbol, count(*) as n
from {{ ref('fct_performance_attribution') }}
group by 1, 2
having count(*) > 1
