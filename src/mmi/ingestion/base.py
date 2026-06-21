"""Extractor base class enforcing a consistent fetch -> validate -> load contract."""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod

import pandas as pd

from mmi.ingestion.loader import DuckDBLoader
from mmi.utils.logging import get_logger
from mmi.utils.redact import redact


class Extractor(ABC):
    """Abstract source extractor.

    Subclasses implement :meth:`fetch`; the base provides validation, loading and
    audit logging so every source behaves identically.
    """

    #: Human-readable source name (used in logs + audit).
    source: str = "base"
    #: Fully-qualified raw table, e.g. ``raw.crypto_prices``.
    table: str = "raw.unknown"
    #: Natural key columns used for idempotent upserts.
    keys: list[str] = []
    #: Required columns the fetched frame must contain.
    required_columns: list[str] = []
    #: If True, a failure here fails the whole run; if False, it's a warn-and-continue source.
    required: bool = True

    def __init__(self, loader: DuckDBLoader) -> None:
        self.loader = loader
        self.log = get_logger(f"ingest.{self.source}")

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Return a dataframe of new/updated rows for this source."""

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Schema-level validation. Override for source-specific rules."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{self.source}: missing columns {missing}")
        df = df.dropna(subset=self.keys)
        return df

    def run(self) -> int:
        """Execute the full pipeline step with audit logging. Returns rows loaded."""
        run_id = self.loader.start_run(self.source)
        try:
            df = self.validate(self.fetch())
            rows = self.loader.upsert(self.table, df, self.keys)
            self.loader.finish_run(run_id, rows, "success")
            return rows
        except Exception as exc:  # noqa: BLE001 - we re-raise after recording
            # Redact before logging/persisting: httpx errors embed the request URL, which for
            # FRED/Gemini carries the API key as a query param.
            self.log.error("extractor failed:\n%s", redact(traceback.format_exc()))
            self.loader.finish_run(run_id, 0, "failed", redact(str(exc)))
            raise
