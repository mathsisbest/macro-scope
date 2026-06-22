-- Contract for the pairwise stats:
--   1. the difference CI is ordered (diff_lo <= diff_hi),
--   2. sharpe_diff reconciles to sharpe_a - sharpe_b, and
--   3. `distinguishable` is exactly "the CI excludes zero".
-- Any violation means the Python-landed row is internally inconsistent. Passes on zero rows.
select strategy_a, strategy_b, diff_lo, diff_hi, sharpe_diff, distinguishable
from {{ ref('fct_portfolio_strategy_pairs') }}
where diff_lo > diff_hi
   or abs(sharpe_diff - (sharpe_a - sharpe_b)) > 1e-9
   or distinguishable <> (diff_lo > 0 or diff_hi < 0)
