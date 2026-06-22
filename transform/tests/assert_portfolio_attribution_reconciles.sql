-- Contract: per strategy, the per-asset return contributions must reconcile to the strategy's
-- gross return (the '(costs)' row is excluded — it carries the separate cost drag). This is the
-- core attribution guarantee: the decomposition adds back up. Passes when this returns zero rows.
with by_strategy as (
    select
        strategy,
        sum(case when symbol <> '(costs)' then contribution_to_return else 0 end) as asset_return_sum,
        max(strategy_gross_return) as gross_return
    from {{ ref('fct_performance_attribution') }}
    group by strategy
)
select strategy, asset_return_sum, gross_return
from by_strategy
where abs(asset_return_sum - gross_return) > 1e-6
