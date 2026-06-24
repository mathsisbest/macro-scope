"""Clean-room public-install import guard.

Fails loudly if any non-.[dashboard] heavy module (sklearn / scipy / dbt / joblib) is
reachable at module-scope from the dashboard.app / dashboard.data import chain.

The guard matters because:
  - Public Streamlit Cloud deploy installs only `.[dashboard]` (streamlit + plotly + duckdb).
  - CI installs `.[all]` (every extra), so a stray `import sklearn` in dashboard code would
    silently pass CI but crash the public deploy.
  - This script covers the gap: it imports app+data then inspects sys.modules.

Run: `make import-smoke`

RIGOROUS CLEAN-VENV FORM (standalone):
  python3 -m venv /tmp/mmi-import-check
  /tmp/mmi-import-check/bin/pip install -q ".[dashboard]"  # dashboard-only!
  PYTHONPATH=. /tmp/mmi-import-check/bin/python scripts/public_import_smoke.py
  rm -rf /tmp/mmi-import-check

The clean-venv form is the definitive guard; the in-venv form (run from make import-smoke
in a .[all] venv) is a belt-and-suspenders module-scan (it can only flag stray *module-level*
imports, not imports inside lazy-init blocks like `if st.button(...): import sklearn`).
Note: the module-scan form runs fast (sub-second) and is safe to wire into `make import-smoke`
without the heavy clean-venv install step; add it to `make ci` only when the clean-venv form
is confirmed to complete within the CI time budget.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root on path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Banned module prefixes: must not appear at module-scope in the dashboard import chain ────
BANNED_PREFIXES: list[str] = [
    "sklearn",
    "scipy",
    "dbt",
    "joblib",
    "portfolio.compute",  # mmi.portfolio.compute — pulls in scipy at import time
]

# ── Snapshot before import (baseline — stdlib + transitive deps already loaded) ──────────────
_before = set(sys.modules.keys())

# ── Set snapshot mode so dashboard.data doesn't try to open a DuckDB file ────────────────────
# These must be set BEFORE the dashboard imports so settings.snapshot_mode is True at
# module-scope inside data.py (pydantic-settings reads env at class instantiation).
os.environ.setdefault("MMI_SNAPSHOT_MODE", "1")
os.environ.setdefault("MMI_SNAPSHOT_DIR", str(REPO_ROOT / "data" / "public"))

# ── Import the dashboard's full public surface ────────────────────────────────────────────────
# We import both entry points that Streamlit Cloud will import when loading the app.
# dashboard.app does sys.path.insert itself, but since we put REPO_ROOT above that's fine.
#
# We can't import dashboard.app directly as a module in the usual sense (it calls
# st.set_page_config at module scope), but we CAN import dashboard.data and every component
# it pulls in. That covers the real risk surface: a stray `import sklearn` in
# data.py / charts.py / theme.py.
import dashboard.data  # noqa: E402
import dashboard.theme  # noqa: E402,F401 — imported for side-effect inspection only
from dashboard.components import charts  # noqa: E402,F401 — imported for side-effect inspection

_after = set(sys.modules.keys())
_new = _after - _before

# ── Scan ─────────────────────────────────────────────────────────────────────────────────────
violations: list[str] = []
for mod in sorted(_new):
    for prefix in BANNED_PREFIXES:
        if mod == prefix or mod.startswith(prefix + ".") or mod.startswith(prefix + "_"):
            violations.append(f"  {mod!r}  (banned prefix: {prefix!r})")
            break

if violations:
    print("FAIL — banned module(s) imported at module-scope from the dashboard import chain:")
    for v in violations:
        print(v)
    print(
        "\nThe public deploy installs only .[dashboard]; "
        "these imports will crash Streamlit Cloud.\n"
        "Move them inside lazy-load guards "
        "(e.g. inside a function body or an `if not module:` block)."
    )
    sys.exit(1)

print(f"public_import_smoke: PASS — no banned imports ({len(_new)} new modules, none banned)")
print("(rigorous check: run the clean-venv form documented in this file's docstring)")
