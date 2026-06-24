-- Singular test: daily simple returns should be within sane bounds (catches bad joins
-- or corrupt source rows). Passes when zero rows are returned.
select symbol, date, daily_return
from {{ ref('fct_asset_daily') }}
where daily_return is not null
  and abs(daily_return) > 0.5  -- a >50% single-day move in a tracked asset is suspicious
  -- No allowlisted exceptions: the universe is broad-market ETFs / FX / BTC, none of which has a
  -- real >50% single-day move. If a verified extreme ever appears, exclude it here (and past a
  -- handful, promote the list to a `known_extreme_events` seed).
