-- Pairwise bootstrap Sharpe-difference CIs (computed by `mmi portfolio`, landed in raw).
-- `distinguishable` is true when the difference CI excludes zero; a singular test re-derives it
-- from the CI bounds so the Python-landed flag can never silently disagree with the numbers.
select
    strategy_a,
    strategy_b,
    sharpe_a,
    sharpe_b,
    sharpe_diff,
    diff_lo,
    diff_hi,
    distinguishable
from {{ source('raw', 'portfolio_strategy_pairs') }}
