-- Contract: every confidence interval must be ordered (lo <= point <= hi is not guaranteed for a
-- percentile bootstrap, but lo <= hi always must hold). Passes when this returns zero rows.
select strategy, sharpe_lo, sharpe_hi
from {{ ref('fct_portfolio_strategy_stats') }}
where sharpe_lo > sharpe_hi
