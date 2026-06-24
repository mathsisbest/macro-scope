"""FRED macro extractor (free API key). Series listed in config/assets.yml."""

from __future__ import annotations

import pandas as pd

from mmi.ingestion.base import Extractor
from mmi.settings import load_assets, settings
from mmi.utils.http import get_json

_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredExtractor(Extractor):
    source = "fred"
    table = "raw.macro_series"
    keys = ["series_id", "date"]
    required_columns = ["series_id", "date", "value"]
    probe_url = _URL

    def probe(self) -> None:
        """Probe FRED with a real key-authenticated request against a well-known series."""
        params = {
            "series_id": "DGS10",
            "api_key": settings.fred_api_key,
            "file_type": "json",
        }
        get_json(_URL, params=params)

    def skip_reason(self) -> str | None:
        # FRED needs a (free) key. Unkeyed, skip gracefully so the keyless core still lands and
        # the scheduled ingest exits 0 — rather than failing the run. A *keyed* FRED failure
        # still fails the run (FRED is the reliable macro core), since ``required`` stays True.
        if not settings.fred_api_key:
            return "FRED_API_KEY not set (get a free key at fredaccount.stlouisfed.org)"
        return None

    def fetch(self) -> pd.DataFrame:
        frames = [self._fetch_series(s["id"]) for s in load_assets()["macro"]]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_series(self, series_id: str) -> pd.DataFrame:
        params = {
            "series_id": series_id,
            "api_key": settings.fred_api_key,
            "file_type": "json",
        }
        obs = get_json(_URL, params=params).get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty:
            return df
        df["series_id"] = series_id
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
        df["source"] = self.source
        return df[["series_id", "date", "value", "source"]]
