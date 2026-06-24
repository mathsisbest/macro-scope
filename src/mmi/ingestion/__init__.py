"""Ingestion layer (data engineering): one Extractor per free data source."""

from mmi.ingestion.base import Extractor
from mmi.ingestion.coingecko import CoinGeckoExtractor
from mmi.ingestion.fred import FredExtractor
from mmi.ingestion.loader import DuckDBLoader
from mmi.ingestion.worldbank import WorldBankExtractor
from mmi.ingestion.yahoo import YahooChartExtractor

#: All registered extractors the CLI iterates.
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
    "FredExtractor",
    "WorldBankExtractor",
    "EXTRACTORS",
]
