"""Decide the dashboard's storage mode *before* the settings singleton is built.

pydantic-settings reads ``MMI_SNAPSHOT_MODE`` from the environment, but Streamlit Community
Cloud exposes secrets via ``st.secrets`` and does not reliably promote them to environment
variables. To keep the public app zero-config (the README's "no secrets required in the public
app"), the dashboard entrypoint calls ``resolve_snapshot_mode`` at startup: when the operator
hasn't pinned a mode and there's no live database to read but the committed Parquet snapshot
exists, it switches snapshot mode on. Kept pure (paths/env injected) so it is unit-testable
without import side effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def resolve_snapshot_mode(environ: Mapping[str, str], repo_root: Path) -> str | None:
    """Return ``"1"`` if the dashboard should default to the committed Parquet snapshot, else
    ``None`` (leave the environment untouched).

    An explicit ``MMI_SNAPSHOT_MODE`` or a configured live store (DuckDB file / MotherDuck) always
    wins — this only fills the gap when neither is present, which is exactly the public-deploy
    case (Streamlit Cloud has the committed ``data/public`` but no ``data/mmi.duckdb`` and no
    MotherDuck token). Local dev is untouched: after ``make demo`` the live DuckDB exists.
    """
    # An explicit choice always wins — never override the operator.
    if environ.get("MMI_SNAPSHOT_MODE") is not None:
        return None
    # A configured MotherDuck target is a live store — don't shadow it with the snapshot.
    if environ.get("MOTHERDUCK_TOKEN") and environ.get("MMI_MOTHERDUCK_DATABASE"):
        return None

    db_override = environ.get("MMI_DUCKDB_PATH")
    live_db = Path(db_override) if db_override else repo_root / "data" / "mmi.duckdb"
    snap_override = environ.get("MMI_SNAPSHOT_DIR")
    snapshot_dir = Path(snap_override) if snap_override else repo_root / "data" / "public"

    snapshot_present = snapshot_dir.is_dir() and any(snapshot_dir.glob("*.parquet"))
    if not live_db.exists() and snapshot_present:
        return "1"
    return None
