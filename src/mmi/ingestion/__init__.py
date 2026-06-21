"""Ingestion layer (data engineering): one Extractor per free data source."""

from mmi.ingestion.base import Extractor
from mmi.ingestion.coingecko import CoinGeckoExtractor
from mmi.ingestion.fred import FredExtractor
from mmi.ingestion.loader import DuckDBLoader
from mmi.ingestion.stooq import StooqExtractor
from mmi.ingestion.worldbank import WorldBankExtractor
from mmi.ingestion.yahoo import YahooChartExtractor

#: All registered extractors the CLI iterates. Yahoo (adjusted close) supplies price history;
#: Stooq is kept as a dormant best-effort fallback (its free CSV endpoint now returns a JS
#: browser-challenge, not CSV) and is intentionally out of the active rotation.
EXTRACTORS: list[type[Extractor]] = [
    CoinGeckoExtractor,
    YahooChartExtractor,
    FredExtractor,
    WorldBankExtractor,
]

__all__ = [
    "Extractor",
    "DuckDBLoader",
    "CoinGeckoExtractor",
    "YahooChartExtractor",
    "StooqExtractor",
    "FredExtractor",
    "WorldBankExtractor",
    "EXTRACTORS",
]
