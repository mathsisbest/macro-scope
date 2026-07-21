"""Ingestion layer (data engineering): one Extractor per free data source."""

from mmi.ingestion.base import Extractor
from mmi.ingestion.fred import FredExtractor
from mmi.ingestion.loader import DuckDBLoader
from mmi.ingestion.odds import OddsExtractor
from mmi.ingestion.worldbank import WorldBankExtractor
from mmi.ingestion.yahoo import YahooChartExtractor

#: All registered extractors the CLI iterates.
EXTRACTORS: list[type[Extractor]] = [
    YahooChartExtractor,
    FredExtractor,
    WorldBankExtractor,
    OddsExtractor,
]

__all__ = [
    "Extractor",
    "DuckDBLoader",
    "YahooChartExtractor",
    "FredExtractor",
    "WorldBankExtractor",
    "OddsExtractor",
    "EXTRACTORS",
]
