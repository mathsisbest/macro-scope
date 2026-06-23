-- Regime-conditional performance: how does each strategy behave in Low / Medium / High market
-- volatility regimes? The regime is SPY's 20-day-vol terciles (the same definition as ml.regime,
-- but derived here in SQL so it needs no ML run), joined to each strategy's daily returns. Pure
-- dbt over two marts — tested SQL, not ad-hoc Python. Grain: (window_id, strategy, regime).
--
-- IMPORTANT: each strategy sits in cash (daily_return = 0) until its first rebalance (~lookback
-- days). Those warm-up zeros are trimmed here so the regime stats are computed over the INVESTED
-- period only — matching the bootstrap (#19) and attribution (#22) marts. Including them would
-- pull every return toward 0 and understate vol (zeros add no variance), distorting Sharpe.
--
-- The SPY terciles are cut PER WINDOW, over each window's own invested span (NTILE partitioned by
-- window_id). So "High vol" is relative to that window's era — regime labels are NOT comparable
-- across windows (a 2015+ "High" is not a 2002+ "High"). This is the right call for a look-ahead-
-- free, window-local regime view; the dashboard documents the non-comparability.
with window_bounds as (
    select window_id, min(date) as lo, max(date) as hi
    from {{ ref('fct_portfolio_returns') }}
    where daily_return <> 0
    group by window_id
),

first_invested as (
    select window_id, strategy, min(date) as start_date
    from {{ ref('fct_portfolio_returns') }}
    where daily_return <> 0
    group by window_id, strategy
),

spy as (
    select date, vol_20d
    from {{ ref('fct_asset_daily') }}
    where symbol = 'SPY' and vol_20d is not null
),

spy_regime as (
    select
        b.window_id,
        s.date,
        case ntile(3) over (partition by b.window_id order by s.vol_20d)
            when 1 then 'Low'
            when 2 then 'Medium'
            else 'High'
        end as regime
    from window_bounds as b
    join spy as s on s.date between b.lo and b.hi
),

joined as (
    select p.window_id, p.strategy, r.regime, p.daily_return
    from {{ ref('fct_portfolio_returns') }} as p
    join first_invested as fi
        on fi.window_id = p.window_id and fi.strategy = p.strategy and p.date >= fi.start_date
    join spy_regime as r on r.window_id = p.window_id and r.date = p.date
)

select
    window_id,
    strategy,
    regime,
    count(*) as n_days,
    count(*) * 1.0 / sum(count(*)) over (partition by window_id, strategy) as day_share,
    avg(daily_return) * 252 as ann_return,
    stddev_samp(daily_return) * sqrt(252) as ann_vol,
    case
        when stddev_samp(daily_return) = 0 then null
        else avg(daily_return) / stddev_samp(daily_return) * sqrt(252)
    end as ann_sharpe
from joined
group by window_id, strategy, regime
