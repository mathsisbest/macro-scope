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
    #: Fully-qualified raw table, e.g. ``raw.asset_prices``.
    table: str = "raw.unknown"
    #: Natural key columns used for idempotent upserts.
    keys: list[str] = []
    #: Required columns the fetched frame must contain.
    required_columns: list[str] = []
    #: If True, a failure here fails the whole run; if False, it's a warn-and-continue source.
    required: bool = True
    #: URL hit by the connectivity probe (no full fetch/parse). Empty string = not configured.
    probe_url: str = ""
    #: Column name to use for incremental watermark (e.g. "date"). None = full refresh.
    watermark_col: str | None = None

    def __init__(self, loader: DuckDBLoader) -> None:
        self.loader = loader
        self.log = get_logger(f"ingest.{self.source}")

    @abstractmethod
    def fetch(self, start_after: str | None = None) -> pd.DataFrame:
        """Return a dataframe of new/updated rows for this source.

        Parameters
        ----------
        start_after:
            If incremental, the latest value of ``watermark_col`` already in the raw table.
            Subclasses use this to fetch only new data (e.g. ``period1=<date>`` for Yahoo).
            None means full refresh.
        """

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Schema-level validation. Override for source-specific rules."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{self.source}: missing columns {missing}")
        df = df.dropna(subset=self.keys)
        return df

    def skip_reason(self) -> str | None:
        """Return a human reason to SKIP this run (e.g. a missing API key), or None to run.

        A skipped run is audited as ``"skipped"`` — deliberately distinct from ``"success"``
        (it never ran) and ``"failed"`` (nothing is broken). This lets a key-gated source (e.g.
        FRED) sit out a *keyless* run so the no-key core (Yahoo, World Bank) still lands and
        ``mmi ingest`` exits 0; adding the key later folds the source back in with no code change.
        """
        return None

    def probe(self) -> None:
        """Lightweight connectivity check for healthcheck.

        Raise on any failure; return None on success.
        Default: a single GET of probe_url via mmi.utils.http.get_json. Subclasses may override.
        MUST NOT call self.fetch() or write to the DB.
        MUST raise if probe_url is empty (no false ok).
        """
        from mmi.utils.http import get_json

        if not self.probe_url:
            raise RuntimeError(f"{self.source}: probe_url is not configured")
        get_json(self.probe_url)

    def run(self) -> int:
        """Execute the full pipeline step with audit logging. Returns rows loaded."""
        run_id = self.loader.start_run(self.source)
        reason = self.skip_reason()
        if reason:
            self.log.warning("%s: skipping — %s", self.source, reason)
            self.loader.finish_run(run_id, 0, "skipped", reason)
            return 0
        try:
            # Incremental watermark: query the latest date in the raw table
            start_after = None
            if self.watermark_col:
                wm = self.loader.watermark(self.table, self.watermark_col)
                if wm:
                    start_after = wm
                    self.log.info("%s: incremental from %s", self.source, start_after)
                else:
                    self.log.info("%s: no existing data — full refresh", self.source)
            df = self.validate(self.fetch(start_after=start_after))
            rows = self.loader.upsert(self.table, df, self.keys)
            self.loader.finish_run(run_id, rows, "success")
            return rows
        except Exception as exc:  # noqa: BLE001 - we re-raise after recording
            # Redact before logging/persisting: httpx errors embed the request URL, which for
            # FRED/Gemini carries the API key as a query param.
            self.log.error("extractor failed:\n%s", redact(traceback.format_exc()))
            try:
                self.loader.finish_run(run_id, 0, "failed", redact(str(exc)))
            except Exception:  # noqa: BLE001 - audit failure must not mask the original error
                self.log.error(
                    "audit write failed after extractor error:\n%s",
                    redact(traceback.format_exc()),
                )
            raise
