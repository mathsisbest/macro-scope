-- Estrella-Mishkin probit recession-risk model (NY Fed calibration).
--
-- P(recession within 12m) = normal_cdf(alpha + beta * spread)
--   alpha = -0.5333, beta = -0.6629  (NY Fed)
--
-- Spread: canonical 10Y-3M (DGS10 - DGS3MO) when DGS3MO is present in the data;
-- fall back to 10Y-2Y (DGS10 - DGS2) when DGS3MO is absent (e.g. synthetic seed).
-- The `model` column records which spread was used.
--
-- DuckDB has no native normal CDF or erf/erfc function. We implement:
--   normal_cdf(x) via Abramowitz & Stegun (1964) §26.2.17 rational approximation.
--   Maximum absolute error: < 7.5e-8 across all x. Suitable for the ~2-3 decimal
--   place precision that matters in practice for the recession probability output.
--
-- A&S 26.2.17 coefficients:
--   p  = 0.2316419
--   b1 = 0.319381530, b2 = -0.356563782, b3 = 1.781477937
--   b4 = -1.821255978, b5 = 1.330274429
-- For x >= 0:  Phi(x) = 1 - phi(x) * (b1*t + b2*t^2 + b3*t^3 + b4*t^4 + b5*t^5)
--              where t = 1/(1 + p*x),  phi(x) = exp(-x^2/2) / sqrt(2*pi)
-- For x < 0:   Phi(x) = 1 - Phi(-x)
--
-- This is a CHEAP DAILY mart — NOT tagged `portfolio` (see dbt_project.yml for tag logic).

with y10 as (
    select date, value as dgs10
    from {{ ref('fct_macro_indicator') }}
    where series_id = 'DGS10'
),

y3mo as (
    select date, value as dgs3mo
    from {{ ref('fct_macro_indicator') }}
    where series_id = 'DGS3MO'
),

y2 as (
    select date, value as dgs2
    from {{ ref('fct_macro_indicator') }}
    where series_id = 'DGS2'
),

-- ASOF JOIN so every DGS10 row carries the latest-available 3-month and 2-year yields,
-- mirroring the approach in fct_market_macro.
joined as (
    select
        y10.date,
        y10.dgs10,
        y3mo.dgs3mo,
        y2.dgs2
    from y10
    asof left join y3mo on y10.date >= y3mo.date
    asof left join y2    on y10.date >= y2.date
),

with_spread as (
    select
        date,
        -- Use 10Y-3M when available; fall back to 10Y-2Y proxy.
        case
            when dgs3mo is not null then dgs10 - dgs3mo
            else                         dgs10 - dgs2
        end  as spread_10y_3m,
        case
            when dgs3mo is not null then '10y_3m'
            else                         '10y_2y_proxy'
        end  as model
    from joined
    where dgs10 is not null
      and (dgs3mo is not null or dgs2 is not null)
),

with_index as (
    select
        date,
        spread_10y_3m,
        model,
        -- NY Fed coefficients: alpha = -0.5333, beta = -0.6629
        -0.5333 + (-0.6629 * spread_10y_3m) as z
    from with_spread
),

-- Compute the A&S 26.2.17 rational approximation of normal_cdf(z).
-- We evaluate Phi for |z| (always >= 0), then mirror for z < 0.
with_prob as (
    select
        date,
        spread_10y_3m,
        model,
        z,
        -- t = 1 / (1 + 0.2316419 * |z|)
        1.0 / (1.0 + 0.2316419 * abs(z))                    as t,
        -- phi(|z|) = exp(-z^2 / 2) / sqrt(2*pi)
        exp(-0.5 * z * z) / sqrt(2.0 * 3.141592653589793)   as pdf
    from with_index
)

select
    date,
    spread_10y_3m,
    -- normal_cdf(z):
    --   Phi(|z|) = 1 - pdf * poly(t)
    --   poly(t)  = b1*t + b2*t^2 + b3*t^3 + b4*t^4 + b5*t^5  (Horner form)
    --   If z < 0: Phi(z) = 1 - Phi(|z|) = pdf * poly(t)
    case
        when z >= 0 then
            1.0 - pdf * (t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))))
        else
                pdf * (t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))))
    end  as recession_prob,
    model
from with_prob
order by date
