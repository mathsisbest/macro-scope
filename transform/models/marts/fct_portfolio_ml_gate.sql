-- The ML gate over time: per rebalance, the forecast's out-of-sample skill vs the historical-mean
-- prior (forecast_skill in [0,1]) and the weight it earns in mvo_ml's mu blend (forecast_weight =
-- lambda). A persistently low forecast_weight is the evidence that mvo_ml ≈ mvo_histmean *because*
-- the forecast has no edge — not a bug. Computed by `mmi portfolio`, landed in raw.
select
    date,
    forecast_skill,
    forecast_weight
from {{ source('raw', 'portfolio_ml_gate') }}
