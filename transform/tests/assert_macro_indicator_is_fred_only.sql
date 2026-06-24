-- Singular test: the macro mart must contain ONLY FRED series. The dashboard's Macro tab shows an
-- unconditional "Source: FRED, Federal Reserve Bank of St. Louis" attribution caption (scope 5 of
-- #51) because every series here is FRED-sourced. If a non-FRED series (e.g. a World Bank indicator)
-- ever lands in this mart, that caption becomes a MISATTRIBUTION — so fail the build and force a
-- revisit of the attribution (add the series here AND gate the caption appropriately).
-- Passes when zero rows are returned.
select distinct series_id
from {{ ref('fct_macro_indicator') }}
where series_id not in ('CPIAUCSL', 'UNRATE', 'DGS10', 'DGS2', 'FEDFUNDS')
