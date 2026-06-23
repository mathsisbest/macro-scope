-- Singular test: the grain of fct_performance_attribution must be one row per
-- (window_id, strategy, symbol), including the '(costs)' pseudo-symbol. Passes on zero rows.
select window_id, strategy, symbol, count(*) as n
from {{ ref('fct_performance_attribution') }}
group by 1, 2, 3
having count(*) > 1
