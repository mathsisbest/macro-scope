"""Tests for the FRED ingestion config and extractor behaviour.

Verifies that:
- All expected FRED series IDs are declared in config/assets.yml, including
  DGS3MO (required for the canonical 10Y-3M recession-risk spread in Wave 2).
- The keyless skip-gracefully behaviour is preserved: FredExtractor.skip_reason()
  returns a non-None string when FRED_API_KEY is absent.
- FredExtractor.skip_reason() returns None when a key is present (run proceeds).
"""

from __future__ import annotations

import pytest

from mmi.settings import load_assets

# ---------------------------------------------------------------------------
# Config-level assertions
# ---------------------------------------------------------------------------

REQUIRED_FRED_SERIES = {
    "CPIAUCSL",  # US CPI
    "UNRATE",  # US Unemployment Rate
    "DGS10",  # 10Y Treasury Yield
    "DGS2",  # 2Y Treasury Yield
    "DGS3MO",  # 3-Month Treasury Yield — enables 10Y-3M spread (Wave 2)
    "FEDFUNDS",  # Effective Fed Funds Rate
}


def _fred_ids() -> set[str]:
    """Return the set of FRED series IDs declared in config/assets.yml."""
    assets = load_assets()
    return {entry["id"] for entry in assets.get("macro", [])}


def test_dgs3mo_is_in_fred_config() -> None:
    """DGS3MO must be present so the 10Y-3M spread can be computed in Wave 2."""
    assert "DGS3MO" in _fred_ids(), (
        "DGS3MO is missing from config/assets.yml macro section. "
        "It is required for the Estrella-Mishkin recession-risk spread (E2)."
    )


def test_all_required_fred_series_present() -> None:
    """All canonical FRED series must be declared; no accidental removals."""
    declared = _fred_ids()
    missing = REQUIRED_FRED_SERIES - declared
    assert not missing, f"FRED series missing from config/assets.yml: {missing}"


# ---------------------------------------------------------------------------
# Extractor behaviour — keyless skip-gracefully
# ---------------------------------------------------------------------------


def test_fred_skips_when_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """FredExtractor must return a skip reason (not None) when FRED_API_KEY is unset."""
    import mmi.ingestion.fred as fred_mod

    monkeypatch.setattr(fred_mod.settings, "fred_api_key", "")
    reason = fred_mod.FredExtractor(loader=None).skip_reason()
    assert reason is not None, "FredExtractor should skip (not fail) when key is absent"
    assert isinstance(reason, str) and len(reason) > 0


def test_fred_runs_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """FredExtractor must return None from skip_reason() when a key is configured."""
    import mmi.ingestion.fred as fred_mod

    monkeypatch.setattr(fred_mod.settings, "fred_api_key", "a-valid-key")
    assert fred_mod.FredExtractor(loader=None).skip_reason() is None
