-- Per-strategy annualised Sharpe + stationary block-bootstrap CI, computed by `mmi portfolio`
-- and landed in raw. dbt declares the source and enforces the contracts via tests (grain,
-- accepted strategies, CI ordering) — the Python↔SQL boundary done right.
select
    window_id,
    strategy,
    sharpe,
    sharpe_lo,
    sharpe_hi,
    n_obs,
    n_boot,
    ci_pct,
    block_days
from {{ source('raw', 'portfolio_strategy_stats') }}
