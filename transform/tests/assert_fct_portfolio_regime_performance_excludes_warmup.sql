-- Contract: the regime mart must cover each strategy's INVESTED period only (post-warm-up),
-- consistent with the bootstrap (#19) and attribution (#22) marts. It fails if a strategy's
-- regime-day total exceeds its invested-day count — which is exactly what happens if the warm-up
-- zero-return days leak back into the regime join. Passes when this query returns zero rows.
with first_invested as (
    select strategy, min(date) as start_date
    from {{ ref('fct_portfolio_returns') }}
    where daily_return <> 0
    group by strategy
),

invested as (
    select p.strategy, count(*) as invested_days
    from {{ ref('fct_portfolio_returns') }} as p
    join first_invested as fi on fi.strategy = p.strategy and p.date >= fi.start_date
    group by p.strategy
),

regime_total as (
    select strategy, sum(n_days) as regime_days
    from {{ ref('fct_portfolio_regime_performance') }}
    group by strategy
)

select i.strategy, i.invested_days, r.regime_days
from invested as i
join regime_total as r on r.strategy = i.strategy
where r.regime_days > i.invested_days
