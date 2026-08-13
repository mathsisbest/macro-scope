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


def verdict_status_message(verdict: dict) -> tuple[str, bool]:
    """One honest skill-gate status line + whether it must render as a warning.

    Returns ``(label, is_warning)``.  The not-cleared (escape-hatch) state is a
    warning so the UI can surface it unmissably; a cleared gate is a neutral
    caption.  The label is sourced ONLY from the skill_verdict() output.
    """
    if verdict["cleared"]:
        r2 = verdict["oos_r2"]
        ratio = verdict["qlike_skill_ratio"]
        folds_passed = verdict["folds_passed"]
        n_folds = verdict["n_folds"]
        return (
            f"Volatility model skill gate: CLEARED — beats the persistence baseline "
            f"out-of-sample (OOS R²={r2:.3f} ≥ 0.10; QLIKE skill ratio={ratio:.3f} < 0.99; "
            f"{folds_passed}/{n_folds} folds passed).",
            False,
        )
    reasons = "; ".join(verdict["reasons"]) if verdict["reasons"] else "metrics not yet available"
    return (
        f"Volatility model skill gate: NOT CLEARED — baseline-only, no demonstrated "
        f"out-of-sample edge. {reasons}",
        True,
    )


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
