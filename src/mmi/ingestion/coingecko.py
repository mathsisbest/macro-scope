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
