-- Singular test: the macro mart must contain ONLY FRED series. This is the precondition for the
-- dashboard's macro source caption (dashboard.data.macro_source_caption / scope 5 of #51), which
-- attributes LIVE macro data to "FRED, Federal Reserve Bank of St. Louis". If a non-FRED series
-- (e.g. a World Bank indicator) ever lands in this mart, that attribution would be a MISATTRIBUTION
-- — so fail the build and force a revisit (add the series here AND adjust the caption logic).
-- Passes when zero rows are returned.
select distinct series_id
from {{ ref('fct_macro_indicator') }}
where series_id not in ('CPIAUCSL', 'UNRATE', 'DGS10', 'DGS2', 'FEDFUNDS')
