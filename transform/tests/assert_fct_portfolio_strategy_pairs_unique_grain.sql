-- Singular test: the grain of fct_portfolio_strategy_pairs must be one row per
-- (window_id, strategy_a, strategy_b). Passes when this query returns zero rows.
-- (Pairs had no grain guard before Phase D; it needs one now that the table carries every window.)
select window_id, strategy_a, strategy_b, count(*) as n
from {{ ref('fct_portfolio_strategy_pairs') }}
group by 1, 2, 3
having count(*) > 1
