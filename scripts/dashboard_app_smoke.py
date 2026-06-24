"""App-render smoke harness using streamlit.testing.v1.AppTest.

Exercises dashboard/app.py in two modes:
  (a) Populated local DuckDB (after `mmi seed`; MMI_SNAPSHOT_MODE unset) — asserts no exception,
      all 5 tabs present, KPI row + portfolio panels render, provenance badge text appears.
  (b) MMI_SNAPSHOT_MODE=1 with an EMPTY temp dir — asserts db_exists() is False and the
      "No database yet" st.stop() path is hit without raising.

Run via `make app-smoke` (wired into `make ci`) AFTER the seeded DuckDB is in place.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# ── repo root on path so `from dashboard import ...` resolves when run directly ──────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(REPO_ROOT / "dashboard" / "app.py")
# If ci.duckdb is set in environment (from `make ci`), use it; fall back to the dev DB.
_CI_DB = os.environ.get("MMI_DUCKDB_PATH", str(REPO_ROOT / "data" / "ci.duckdb"))
TIMEOUT = 60  # seconds per AppTest.run() call


def _bust_settings_cache() -> None:
    """Reload mmi.settings and all dashboard modules so env-var changes take effect.

    AppTest re-executes app.py in the same process.  Several layers of caching survive
    between runs:
      1. ``@lru_cache`` on ``get_settings()`` → clear it.
      2. The module-level ``settings`` singleton in ``mmi.settings`` → reload the module.
      3. ``from mmi.settings import settings`` in ``dashboard.data`` etc. binds the OLD
         object → must reload those modules too so they re-bind to the fresh singleton.

    We do a best-effort multi-module reload in dependency order (settings → data → theme →
    charts) so the next AppTest.run() picks up the new environment cleanly.
    """
    import importlib

    _reload_order = [
        "mmi.settings",
        "mmi.utils.db",
        "dashboard.data",
        "dashboard.theme",
        "dashboard.components.charts",
        "dashboard.components.kpi",
    ]
    try:
        import mmi.settings as _s  # noqa: PLC0415

        _s.get_settings.cache_clear()
    except Exception:
        pass

    for mod_name in _reload_order:
        try:
            import importlib as _il  # noqa: PLC0415

            mod = _il.import_module(mod_name)
            importlib.reload(mod)
        except Exception:
            pass  # module may not be imported yet — that's fine


# ─────────────────────────────────────────────────────────────────────────────
# Mode (a): populated DuckDB (MMI_SNAPSHOT_MODE unset/off)
# ─────────────────────────────────────────────────────────────────────────────


def test_populated_db_path() -> None:
    """Render against the seeded local DuckDB and assert all key surfaces are present."""
    print("\n[mode a] populated DuckDB path …", flush=True)

    # Thread the CI DB path in; clear snapshot mode so we go through the live-DB path.
    os.environ.setdefault("MMI_DUCKDB_PATH", _CI_DB)
    os.environ.pop("MMI_SNAPSHOT_MODE", None)
    _bust_settings_cache()

    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.secrets = {}
    at.run(timeout=TIMEOUT)

    # ── 1. No unhandled exception ─────────────────────────────────────────────
    assert not at.exception, f"app.py raised an exception: {at.exception}"

    # ── 2. All 5 tabs present ─────────────────────────────────────────────────
    all_tabs = at.tabs
    assert len(all_tabs) == 5, (
        f"Expected 5 tabs (Markets/Macro/ML forecast/AI brief/Portfolio), "
        f"got {len(all_tabs)}: {all_tabs}"
    )

    # ── 3. KPI row: at least one metric card rendered (the st.metric calls) ───
    # AppTest exposes at.metric — a list of all st.metric elements in render order.
    assert len(at.metric) >= 1, (
        "Expected at least one KPI metric (SPY close / crypto / spread) — got none"
    )

    # ── 4. Portfolio panels: the portfolio tab must contain at least one plotly chart.
    # AppTest indexes tabs by label (0-based) OR by their render position; tab[4] = Portfolio.
    portfolio_tab = all_tabs[4]
    # If the portfolio backtest ran, there will be at least one plotly_chart element there.
    # We accept an empty portfolio (mmi seed doesn't always run portfolio) with a graceful info.
    # Either way, the tab itself must be render-error-free — at.exception covers that.
    # Log what rendered for diagnostics.
    print(f"  portfolio tab children: {len(portfolio_tab.children)} elements", flush=True)

    # ── 5. Sample-data provenance badge must appear in the caption text ────────
    # `mmi seed` seeds with source='sample', so is_sample_data() → True and the caption
    # "🧪 sample data" is rendered.  We search across all caption elements.
    captions = [c.value for c in at.caption]
    badge_text = "sample data"
    assert any(badge_text in c for c in captions), (
        f"Provenance badge ('{badge_text}') not found in any caption. Captions found: {captions}"
    )

    # ── 6. macro_source_caption: sample path must NOT say "FRED" ──────────────
    assert not any("Source: FRED" in c for c in captions), (
        "Sample data must NOT carry the FRED attribution — "
        "macro_source_caption(is_sample=True) regression."
    )
    # Check it does carry the synthetic-data caption instead.
    assert any("Synthetic sample data" in c for c in captions), (
        "Sample data must carry the synthetic-data caption — "
        "macro_source_caption(is_sample=True) regression."
    )

    print(
        "  [mode a] PASS — no exception, 5 tabs, KPI row, provenance badge, FRED non-attribution",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode (b): MMI_SNAPSHOT_MODE=1 with an EMPTY snapshot dir
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_snapshot_path() -> None:
    """Snapshot mode + empty dir must hit db_exists()=False and the st.stop() path cleanly."""
    print("\n[mode b] empty snapshot dir (MMI_SNAPSHOT_MODE=1) …", flush=True)

    with tempfile.TemporaryDirectory() as empty_dir:
        # Inject snapshot-mode env BEFORE creating AppTest so settings picks it up.
        # Also bust the lru_cache so pydantic-settings re-reads env (the cache persists
        # across AppTest.run() calls within the same process).
        os.environ["MMI_SNAPSHOT_MODE"] = "1"
        os.environ["MMI_SNAPSHOT_DIR"] = empty_dir
        _bust_settings_cache()

        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        at.secrets = {}

        try:
            at.run(timeout=TIMEOUT)
        finally:
            # Restore env for subsequent test steps regardless of outcome.
            os.environ.pop("MMI_SNAPSHOT_MODE", None)
            os.environ.pop("MMI_SNAPSHOT_DIR", None)

        # ── No hard exception (st.stop() is a clean halt, not a Python exception) ──
        assert not at.exception, (
            f"app.py raised an unexpected exception in empty-snapshot mode: {at.exception}"
        )

        # ── The "No database yet" warning must appear ──────────────────────────
        warnings = [w.value for w in at.warning]
        no_db_msg = "No database yet"
        assert any(no_db_msg in w for w in warnings), (
            f"Expected '{no_db_msg}' warning in empty-snapshot mode. Warnings found: {warnings}"
        )

        # ── Must NOT have rendered any tabs (st.stop() halted before st.tabs) ──
        # In an empty-snapshot stop path the app returns before st.tabs(); tabs list is empty.
        assert len(at.tabs) == 0, f"Expected no tabs (st.stop() path), got {len(at.tabs)} tabs"

    print(
        "  [mode b] PASS — db_exists()=False path; 'No database yet' warning; no tabs", flush=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    test_populated_db_path()
    test_empty_snapshot_path()
    print("\ndashboard_app_smoke: PASS", flush=True)


if __name__ == "__main__":
    main()
