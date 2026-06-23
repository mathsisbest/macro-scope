-- Singular test: daily simple returns should be within sane bounds (catches bad joins
-- or corrupt source rows). Passes when zero rows are returned.
select symbol, date, daily_return
from {{ ref('fct_asset_daily') }}
where daily_return is not null
  and abs(daily_return) > 0.5  -- a >50% single-day move in a tracked asset is suspicious
  -- Verified real extremes (not corruption) — excluded so the guard stays strict elsewhere.
  -- If this list grows past a handful, promote it to a `known_extreme_events` seed.
  and not (symbol = 'AAPL' and date = date '2000-09-29')  -- Apple Sep-2000 profit-warning crash, ~-52% (real)
