-- Singular test: the grain of fct_portfolio_strategy_stats must be one row per
-- (window_id, strategy). Passes when this query returns zero rows.
-- Replaces the single-column `unique` test on strategy, which Phase D's window dimension made
-- legitimately repeat across windows.
select window_id, strategy, count(*) as n
from {{ ref('fct_portfolio_strategy_stats') }}
group by 1, 2
having count(*) > 1
