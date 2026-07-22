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
    watermark_col: str = "date"

    def probe(self) -> None:
        """Probe World Bank API with a single-row request."""
        get_json(self.probe_url, params={"format": "json", "per_page": 1})

    def fetch(self, start_after: str | None = None) -> pd.DataFrame:
        frames = [self._fetch_indicator(i["id"], start_after) for i in load_assets()["worldbank"]]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_indicator(self, indicator: str, start_after: str | None = None) -> pd.DataFrame:
        url = _URL.format(countries=_COUNTRIES, indicator=indicator)
        params = {"format": "json", "per_page": 20000}
        if start_after:
            # World Bank API uses date=YYYY:YYYY range; fetch from year after last known
            try:
                start_year = int(start_after[:4]) + 1
                params["date"] = f"{start_year}:2050"
            except (ValueError, IndexError):
                pass  # Full refresh if date parsing fails
        payload = get_json(url, params=params)
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame()
        rows = [
            {
                "country_id": row["countryiso3code"],
                "indicator_id": indicator,
                "country": row["countryiso3code"],
                "date": str(row["date"]),  # year string; staging casts
                "value": float(row["value"]) if row["value"] is not None else None,
                "source": self.source,
            }
            for row in payload[1]
            if row.get("value") is not None
        ]
        return pd.DataFrame(rows)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        from mmi.ingestion.models import WorldBankIndicatorRow

        df = super().validate(df)
        return self.validate_pydantic(df, WorldBankIndicatorRow)
