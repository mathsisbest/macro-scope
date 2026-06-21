"""Ingestion layer (data engineering): one Extractor per free data source."""

from mmi.ingestion.base import Extractor
from mmi.ingestion.coingecko import CoinGeckoExtractor
from mmi.ingestion.fred import FredExtractor
from mmi.ingestion.loader import DuckDBLoader
from mmi.ingestion.stooq import StooqExtractor
from mmi.ingestion.worldbank import WorldBankExtractor

#: All registered extractors. The CLI iterates this list.
EXTRACTORS: list[type[Extractor]] = [
    CoinGeckoExtractor,
    StooqExtractor,
    FredExtractor,
    WorldBankExtractor,
]

__all__ = [
    "Extractor",
    "DuckDBLoader",
    "CoinGeckoExtractor",
    "StooqExtractor",
    "FredExtractor",
    "WorldBankExtractor",
    "EXTRACTORS",
]
