-- Singular test: the macro mart must contain ONLY FRED series. This is the precondition for the
-- dashboard's macro source caption (dashboard.data.macro_source_caption / scope 5 of #51), which
-- attributes LIVE macro data to "FRED, Federal Reserve Bank of St. Louis". If a non-FRED series
-- (e.g. a World Bank indicator) ever lands in this mart, that attribution would be a MISATTRIBUTION
-- — so fail the build and force a revisit (add the series here AND adjust the caption logic).
-- Passes when zero rows are returned.
-- Allowlist = every FRED series id in config/assets.yml `macro:`. All are genuine FRED series, so
-- the "Source: FRED" attribution stays valid. Keep this in sync when adding/removing macro series.
select distinct series_id
from {{ ref('fct_macro_indicator') }}
where series_id not in (
    -- Growth & activity
    'A191RL1Q225SBEA', 'INDPRO', 'RSAFS', 'UMCSENT',
    -- Inflation
    'CPIAUCSL', 'PCEPILFE',
    -- Labor
    'UNRATE', 'PAYEMS', 'ICSA',
    -- Rates & curve
    'DGS10', 'DGS2', 'DGS3MO', 'T10Y2Y', 'FEDFUNDS',
    -- Fiscal
    'GFDEGDQ188S',
    -- Money & liquidity
    'M2SL', 'WALCL',
    -- Risk & conditions
    'VIXCLS', 'NFCI', 'SAHMREALTIME',
    -- Commodities & FX
    'DCOILWTICO', 'DTWEXBGS'
)
