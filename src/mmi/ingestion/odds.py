"""Prediction market and macroeconomic event odds extractor."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mmi.ingestion.base import Extractor
from mmi.ingestion.models import OddsRow


class OddsExtractor(Extractor):
    """Ingests prediction market and macro event odds into raw.odds_data."""

    source = "odds"
    table = "raw.odds_data"
    keys = ["event_id", "market_id", "timestamp"]
    required_columns = [
        "event_id",
        "market_id",
        "title",
        "outcome",
        "implied_probability",
        "decimal_odds",
        "timestamp",
    ]
    required = False

    def fetch(self, start_after: str | None = None) -> pd.DataFrame:
        """Fetch sample prediction market odds for macro events & market rates."""
        now_iso = datetime.now(timezone.utc).isoformat()
        sample_data = [
            {
                "event_id": "FED_2026_09",
                "market_id": "CUT_25BPS",
                "title": "FOMC Rate Decision September 2026",
                "outcome": "25bps Cut",
                "implied_probability": 0.65,
                "decimal_odds": 1.54,
                "timestamp": now_iso,
            },
            {
                "event_id": "FED_2026_09",
                "market_id": "PAUSE",
                "title": "FOMC Rate Decision September 2026",
                "outcome": "Hold / Pause",
                "implied_probability": 0.30,
                "decimal_odds": 3.33,
                "timestamp": now_iso,
            },
            {
                "event_id": "CPI_2026_Q3",
                "market_id": "CPI_BELOW_2_5",
                "title": "US CPI Inflation Q3 2026",
                "outcome": "Below 2.5% YoY",
                "implied_probability": 0.58,
                "decimal_odds": 1.72,
                "timestamp": now_iso,
            },
            {
                "event_id": "US_RECESSION_2026",
                "market_id": "RECESSION_NO",
                "title": "NBER US Recession in 2026",
                "outcome": "No Recession",
                "implied_probability": 0.82,
                "decimal_odds": 1.22,
                "timestamp": now_iso,
            },
        ]
        df = pd.DataFrame(sample_data)
        return self.validate_pydantic(df, OddsRow)
