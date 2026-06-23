-- Singular test: the grain of fct_portfolio_ml_gate must be one row per (window_id, date).
-- Passes when this query returns zero rows. Replaces the single-column `unique` test on date,
-- which Phase D's window dimension made legitimately repeat across windows.
select window_id, date, count(*) as n
from {{ ref('fct_portfolio_ml_gate') }}
group by 1, 2
having count(*) > 1
