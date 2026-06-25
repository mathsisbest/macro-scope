"""resolve_snapshot_mode: the zero-config snapshot fallback for the public dashboard deploy."""

from __future__ import annotations

from pathlib import Path

from dashboard.snapshot_boot import resolve_snapshot_mode


def _make_snapshot(tmp_path: Path) -> Path:
    """Create data/public with one parquet — presence is all the resolver checks."""
    d = tmp_path / "data" / "public"
    d.mkdir(parents=True)
    (d / "dim_asset.parquet").write_bytes(b"PAR1")
    return d


def test_enables_snapshot_when_no_live_db_but_snapshot_present(tmp_path):
    _make_snapshot(tmp_path)  # no data/mmi.duckdb under tmp_path, no env overrides
    assert resolve_snapshot_mode({}, tmp_path) == "1"


def test_no_change_when_live_db_exists(tmp_path):
    _make_snapshot(tmp_path)
    (tmp_path / "data" / "mmi.duckdb").write_bytes(b"x")
    assert resolve_snapshot_mode({}, tmp_path) is None


def test_explicit_mode_is_never_overridden(tmp_path):
    _make_snapshot(tmp_path)
    assert resolve_snapshot_mode({"MMI_SNAPSHOT_MODE": "0"}, tmp_path) is None
    assert resolve_snapshot_mode({"MMI_SNAPSHOT_MODE": "1"}, tmp_path) is None


def test_motherduck_target_is_not_shadowed(tmp_path):
    _make_snapshot(tmp_path)
    env = {"MOTHERDUCK_TOKEN": "t", "MMI_MOTHERDUCK_DATABASE": "mmi"}
    assert resolve_snapshot_mode(env, tmp_path) is None


def test_no_snapshot_present_leaves_env_untouched(tmp_path):
    # snapshot dir absent → nothing to fall back to
    assert resolve_snapshot_mode({}, tmp_path) is None


def test_respects_path_overrides(tmp_path):
    snap = tmp_path / "elsewhere"
    snap.mkdir()
    (snap / "x.parquet").write_bytes(b"x")
    db = tmp_path / "custom.duckdb"  # does not exist yet
    env = {"MMI_SNAPSHOT_DIR": str(snap), "MMI_DUCKDB_PATH": str(db)}
    assert resolve_snapshot_mode(env, tmp_path) == "1"
    db.write_bytes(b"x")  # now the overridden live DB exists → it wins
    assert resolve_snapshot_mode(env, tmp_path) is None
