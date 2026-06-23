-- The BTC effect per strategy: Sharpe(inc_btc_2015) − Sharpe(ex_btc_2015) with a PAIRED
-- cross-window block-bootstrap CI (computed by `mmi portfolio`, landed in raw). Because the two
-- 2015 windows are period-identical (asserted by assert_portfolio_windows_period_aligned), the
-- difference is paired: the same resampled dates feed both windows, so distinguishable = the
-- difference CI excludes zero is an honest significance verdict, not two independent CIs combined.
-- Grain: one row per strategy. A singular test re-derives the consistency of diff + distinguishable.
select
    strategy,
    sharpe_ex,
    sharpe_inc,
    sharpe_diff,
    diff_lo,
    diff_hi,
    distinguishable,
    n_obs,
    n_boot,
    ci_pct
from {{ source('raw', 'portfolio_btc_effect') }}
