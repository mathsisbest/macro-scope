-- Contract for the BTC-effect mart: (1) the difference CI is ordered (diff_lo <= diff_hi),
-- (2) sharpe_diff reconciles to sharpe_inc - sharpe_ex, and (3) `distinguishable` is exactly
-- "the difference CI excludes zero". Any violation means the Python-landed row is internally
-- inconsistent. Passes on zero rows. (Mirrors assert_portfolio_strategy_pairs_consistent.)
select strategy, sharpe_ex, sharpe_inc, sharpe_diff, diff_lo, diff_hi, distinguishable
from {{ ref('fct_portfolio_btc_effect') }}
where diff_lo > diff_hi
   or abs(sharpe_diff - (sharpe_inc - sharpe_ex)) > 1e-9
   or distinguishable <> (diff_lo > 0 or diff_hi < 0)
