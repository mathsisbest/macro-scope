-- Contract: the regime mart must cover each strategy's INVESTED period only (post-warm-up),
-- consistent with the bootstrap (#19) and attribution (#22) marts. It fails if a strategy's
-- regime-day total exceeds its invested-day count — which is exactly what happens if the warm-up
-- zero-return days leak back into the regime join. Passes when this query returns zero rows.
-- Every CTE keys on (window_id, strategy): without window_id the day-counts of different windows
-- would sum together and a per-window leak could pass undetected (a false green).
with first_invested as (
    select window_id, strategy, min(date) as start_date
    from {{ ref('fct_portfolio_returns') }}
    where daily_return <> 0
    group by window_id, strategy
),

invested as (
    select p.window_id, p.strategy, count(*) as invested_days
    from {{ ref('fct_portfolio_returns') }} as p
    join first_invested as fi
        on fi.window_id = p.window_id and fi.strategy = p.strategy and p.date >= fi.start_date
    group by p.window_id, p.strategy
),

regime_total as (
    select window_id, strategy, sum(n_days) as regime_days
    from {{ ref('fct_portfolio_regime_performance') }}
    group by window_id, strategy
)

select i.window_id, i.strategy, i.invested_days, r.regime_days
from invested as i
join regime_total as r on r.window_id = i.window_id and r.strategy = i.strategy
where r.regime_days > i.invested_days
