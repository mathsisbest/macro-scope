"""Pure helpers extracted from the dashboard entrypoint — testable without Streamlit."""

from __future__ import annotations

import pandas as pd


def provenance_badge(as_of_val: str | None, is_sample_val: bool | None) -> str:
    """Build a human-readable provenance badge string."""
    parts: list[str] = []
    if as_of_val:
        parts.append(f"📅 Data as of **{as_of_val}**")
    if is_sample_val is True:
        parts.append("🧪 sample data (synthetic — run `mmi ingest` for live)")
    elif is_sample_val is False:
        parts.append("🟢 live data")
    elif as_of_val:
        parts.append("⚠️ mixed/unrecorded data provenance")
    return " · ".join(parts) if parts else ""


def sidebar_status(is_sample_val: bool | None, as_of_val: str | None, runs: pd.DataFrame) -> str:
    """Return a human-readable pipeline-health caption."""
    if not runs.empty:
        return ""
    if is_sample_val is True:
        return "Sample data seeded (synthetic; no live ingestion runs)."
    if is_sample_val is False:
        return "Live data from a committed snapshot (no in-app ingestion log)."
    if as_of_val:
        return "Mixed or unrecorded data provenance."
    return "No data yet — run `make demo` or `mmi ingest`."
