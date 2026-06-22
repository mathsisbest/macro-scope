-- Per-(strategy, symbol) performance attribution, computed by `mmi portfolio` and landed in raw.
-- contribution_to_return sums (over asset rows) to strategy_gross_return; a '(costs)' row carries
-- the cost drag. contribution_to_risk is the asset's share of realised variance (sums to 1 over
-- assets). A singular test asserts the return reconciliation.
select
    strategy,
    symbol,
    contribution_to_return,
    contribution_to_risk,
    strategy_gross_return
from {{ source('raw', 'portfolio_attribution') }}
