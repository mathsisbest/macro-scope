-- Contract: per (window_id, strategy), the per-asset return contributions must reconcile to the
-- strategy's gross return (the '(costs)' row is excluded — it carries the separate cost drag). This
-- is the core attribution guarantee: the decomposition adds back up. Passes on zero rows.
-- Grouping by (window_id, strategy) is essential: without window_id, two windows' contributions and
-- gross returns would be summed/max'd together and a per-window decomposition error could hide.
with by_strategy as (
    select
        window_id,
        strategy,
        sum(case when symbol <> '(costs)' then contribution_to_return else 0 end) as asset_return_sum,
        max(strategy_gross_return) as gross_return
    from {{ ref('fct_performance_attribution') }}
    group by window_id, strategy
)
select window_id, strategy, asset_return_sum, gross_return
from by_strategy
where abs(asset_return_sum - gross_return) > 1e-6
