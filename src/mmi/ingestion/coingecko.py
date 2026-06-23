"""CoinGecko crypto extractor (free Demo API: 100 calls/min, 10k/month)."""

from __future__ import annotations

import pandas as pd

from mmi.ingestion.base import Extractor
from mmi.settings import load_assets, settings
from mmi.utils.http import get_json

_URL = "https://api.coingecko.com/api/v3/simple/price"


class CoinGeckoExtractor(Extractor):
    source = "coingecko"
    table = "raw.crypto_prices"
    keys = ["symbol", "ts"]
    required_columns = ["symbol", "ts", "price_usd"]
    # The Demo API works WITHOUT a key (free tier, 100 calls/min), so we always attempt it and
    # land crypto when we can — no skip_reason. But it's an unofficial, rate-limited free endpoint,
    # so a failure (rate-limit / network) is non-fatal: the scheduled ingest still exits 0 on the
    # macro/equity core. A key (COINGECKO_API_KEY) only raises the limits; it isn't required.
    required = False

    def fetch(self) -> pd.DataFrame:
        ids = load_assets()["crypto"]
        headers = (
            {"x-cg-demo-api-key": settings.coingecko_api_key} if settings.coingecko_api_key else {}
        )
        params = {
            "ids": ",".join(ids),
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_last_updated_at": "true",
        }
        data = get_json(_URL, params=params, headers=headers)
        rows = [
            {
                "symbol": coin,
                "ts": pd.to_datetime(payload.get("last_updated_at"), unit="s", utc=True),
                "price_usd": payload.get("usd"),
                "market_cap": payload.get("usd_market_cap"),
                "volume_24h": payload.get("usd_24h_vol"),
                "source": self.source,
            }
            for coin, payload in data.items()
        ]
        return pd.DataFrame(rows)
