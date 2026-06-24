"""World Bank extractor for global macro context — no API key required."""

from __future__ import annotations

import pandas as pd

from mmi.ingestion.base import Extractor
from mmi.settings import load_assets
from mmi.utils.http import get_json

_COUNTRIES = "USA;GBR;WLD"  # United States, United Kingdom, World aggregate
_URL = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"


class WorldBankExtractor(Extractor):
    source = "worldbank"
    table = "raw.worldbank"
    keys = ["indicator_id", "country", "date"]
    required_columns = ["indicator_id", "country", "date", "value"]
    probe_url = "https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.CD"

    def probe(self) -> None:
        """Probe World Bank API with a single-row request."""
        get_json(self.probe_url, params={"format": "json", "per_page": 1})

    def fetch(self) -> pd.DataFrame:
        frames = [self._fetch_indicator(i["id"]) for i in load_assets()["worldbank"]]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_indicator(self, indicator: str) -> pd.DataFrame:
        url = _URL.format(countries=_COUNTRIES, indicator=indicator)
        payload = get_json(url, params={"format": "json", "per_page": 20000})
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame()
        rows = [
            {
                "indicator_id": indicator,
                "country": row["countryiso3code"],
                "date": row["date"],  # year string; staging casts
                "value": row["value"],
                "source": self.source,
            }
            for row in payload[1]
        ]
        return pd.DataFrame(rows)
