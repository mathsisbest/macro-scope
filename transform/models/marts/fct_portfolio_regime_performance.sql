-- Regime-conditional performance: how does each strategy behave in Low / Medium / High market
-- volatility regimes? The regime is SPY's 20-day-vol terciles (the same definition as ml.regime,
-- but derived here in SQL so it needs no ML run), joined to each strategy's daily returns. Pure
-- dbt over two marts — tested SQL, not ad-hoc Python. Grain: (strategy, regime).
--
-- IMPORTANT: each strategy sits in cash (daily_return = 0) until its first rebalance (~lookback
-- days). Those warm-up zeros are trimmed here so the regime stats are computed over the INVESTED
-- period only — matching the bootstrap (#19) and attribution (#22) marts. Including them would
-- pull every return toward 0 and understate vol (zeros add no variance), distorting Sharpe.
with first_invested as (
    select strategy, min(date) as start_date
    from {{ ref('fct_portfolio_returns') }}
    where daily_return <> 0
    group by strategy
),

spy_regime as (
    select
        date,
        case ntile(3) over (order by vol_20d)
            when 1 then 'Low'
            when 2 then 'Medium'
            else 'High'
        end as regime
    from {{ ref('fct_asset_daily') }}
    where symbol = 'SPY' and vol_20d is not null
),

joined as (
    select p.strategy, r.regime, p.daily_return
    from {{ ref('fct_portfolio_returns') }} as p
    join first_invested as fi on fi.strategy = p.strategy and p.date >= fi.start_date
    join spy_regime as r on r.date = p.date
)

select
    strategy,
    regime,
    count(*) as n_days,
    count(*) * 1.0 / sum(count(*)) over (partition by strategy) as day_share,
    avg(daily_return) * 252 as ann_return,
    stddev_samp(daily_return) * sqrt(252) as ann_vol,
    case
        when stddev_samp(daily_return) = 0 then null
        else avg(daily_return) / stddev_samp(daily_return) * sqrt(252)
    end as ann_sharpe
from joined
group by strategy, regime
