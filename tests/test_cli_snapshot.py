"""`mmi snapshot` exports every marts table to Parquet — the public demo's static, secret-free
data source. Exporting the whole schema means new marts are picked up automatically.

Also covers the MMI_PORTFOLIO_N_BOOT env knob wired into cmd_portfolio (task D1).
Also covers manifest writing, per-file atomicity, and daily-cron preservation invariant (task D7).
"""

import argparse
import contextlib
import json
import os
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pandas as pd
import pytest

import mmi.cli as cli
import mmi.settings as settings_mod


def _connect_to(db):
    # Mirror connect(read_only=...) so cmd_snapshot's read-only open works against the fixture.
    return lambda *a, **k: duckdb.connect(str(db), read_only=k.get("read_only", False))


def _marts_db(path) -> None:
    con = duckdb.connect(str(path))
    con.execute("create schema if not exists marts")
    con.execute(
        "create table marts.dim_asset as "
        "select * from (values ('SPY', 'equities'), ('BTC', 'crypto')) t(symbol, asset_class)"
    )
    con.execute(
        "create table marts.fct_portfolio_returns as select * from (values "
        "('ex_btc_2002', 'equal_weight', DATE '2020-01-01', 0.01)) "
        "t(window_id, strategy, date, daily_return)"
    )
    con.close()


def test_snapshot_exports_every_marts_table(monkeypatch, tmp_path):
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))

    assert cli.cmd_snapshot(argparse.Namespace()) == 0

    # one Parquet per marts table, readable, with the right rows
    assert {p.name for p in out.glob("*.parquet")} == {
        "dim_asset.parquet",
        "fct_portfolio_returns.parquet",
    }
    rt = duckdb.connect()
    try:
        n = rt.execute(f"select count(*) from '{out / 'dim_asset.parquet'}'").fetchone()[0]
    finally:
        rt.close()
    assert n == 2  # round-trips


def test_snapshot_with_no_marts_is_a_noop(monkeypatch, tmp_path):
    db = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists marts")
    con.close()
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))

    assert cli.cmd_snapshot(argparse.Namespace()) == 0
    assert not list(out.glob("*.parquet"))  # nothing exported, no crash


# ---------------------------------------------------------------------------
# MMI_PORTFOLIO_N_BOOT env knob (task D1)
# ---------------------------------------------------------------------------


def _minimal_asset_daily(window_id: str = "ex_btc_2002") -> pd.DataFrame:
    """50 rows with a single strategy — enough for bootstrap_strategy_stats (needs ≥2)."""
    dates = pd.bdate_range("2020-01-01", periods=50)
    return pd.DataFrame(
        {
            "window_id": window_id,
            "strategy": "equal_weight",
            "date": dates,
            "daily_return": np.random.default_rng(0).normal(0.0003, 0.01, 50),
        }
    )


def _make_portfolio_mocks(captured: dict):
    """Return a dict of patches that short-circuit the heavy compute pipeline.

    ``captured`` is mutated in-place: ``captured['n_boot_stats']`` / ``captured['n_boot_btc']`` get
    the n_boot value passed to bootstrap_strategy_stats / paired_btc_effect respectively. The BTC
    path is exercised (see the non-empty ``btc_aligned_returns`` mock) so both knobs are captured.
    """
    minimal = _minimal_asset_daily()

    import mmi.portfolio.stats as _stats

    real_bootstrap = _stats.bootstrap_strategy_stats

    def fake_bootstrap(returns_long, *, window="ex_btc_2002", n_boot=2000, **kw):
        captured["n_boot_stats"] = n_boot
        # Delegate to the real implementation with n_boot so the return shape is correct.
        return real_bootstrap(returns_long, window=window, n_boot=n_boot, **kw)

    real_btc_effect = _stats.paired_btc_effect

    def fake_btc_effect(ex_df, inc_df, *, n_boot=2000, **kw):
        captured["n_boot_btc"] = n_boot
        return real_btc_effect(ex_df, inc_df, n_boot=n_boot, **kw)

    compute_patches = {
        "mmi.portfolio.compute.compute_ml_mu_panel": MagicMock(
            return_value=(pd.DataFrame(), pd.DataFrame())
        ),
        "mmi.portfolio.compute.compute_portfolio_returns": MagicMock(return_value=minimal),
        "mmi.portfolio.compute.compute_attribution": MagicMock(
            return_value=pd.DataFrame(columns=["window_id", "strategy", "symbol"])
        ),
        # Non-empty so a btc_floor is derived and the two 2015 windows (hence paired_btc_effect)
        # actually run — otherwise the BTC path is skipped and n_boot_btc is never captured.
        "mmi.portfolio.compute.btc_aligned_returns": MagicMock(
            return_value=pd.DataFrame(
                {"date": [pd.Timestamp("2015-01-02")], "daily_return": [0.01]}
            )
        ),
        "mmi.portfolio.compute.window_asset_daily": MagicMock(return_value=minimal),
        "mmi.portfolio.stats.bootstrap_strategy_stats": fake_bootstrap,
        "mmi.portfolio.stats.paired_btc_effect": fake_btc_effect,
        "mmi.ingestion.loader.reset_portfolio_raw_tables": MagicMock(),
        "mmi.settings.load_assets": MagicMock(return_value={}),
    }
    return compute_patches


def _run_cmd_portfolio_with_nboot(monkeypatch, tmp_path, n_boot_env: str | None) -> dict:
    """Run cmd_portfolio with heavy compute mocked; return captured dict."""
    db = tmp_path / "p.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists raw")
    con.execute("create schema if not exists marts")
    # fct_asset_daily must exist so cmd_portfolio's SELECT succeeds.
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        "('SPY', DATE '2020-01-01', 0.001, 'equities')) "
        "t(symbol, date, daily_return, asset_class)"
    )
    con.close()

    captured: dict = {}
    patches = _make_portfolio_mocks(captured)

    env_ctx = {"MMI_PORTFOLIO_N_BOOT": n_boot_env} if n_boot_env is not None else {}
    # Strip the env var when testing the default so a stale env from a parent process can't bleed.
    env_strip = n_boot_env is None

    with patch.dict(os.environ, env_ctx, clear=False):
        if env_strip:
            os.environ.pop("MMI_PORTFOLIO_N_BOOT", None)
        ctx_managers = [patch(k, v) for k, v in patches.items()]
        # Also patch connect() so cmd_portfolio opens our fixture DB.
        ctx_managers.append(patch.object(cli, "connect", _connect_to(db)))
        for cm in ctx_managers:
            cm.__enter__()
        try:
            cli.cmd_portfolio(argparse.Namespace())
        finally:
            for cm in reversed(ctx_managers):
                with contextlib.suppress(Exception):
                    cm.__exit__(None, None, None)

    return captured


def test_portfolio_nboot_knob_env_var_is_honoured(monkeypatch, tmp_path):
    """A valid MMI_PORTFOLIO_N_BOOT is read and threaded into BOTH bootstrap functions."""
    captured = _run_cmd_portfolio_with_nboot(monkeypatch, tmp_path, "42")
    assert captured.get("n_boot_stats") == 42, (
        f"Expected n_boot=42 from env, got {captured.get('n_boot_stats')}"
    )
    assert captured.get("n_boot_btc") == 42, (
        f"Expected n_boot=42 into paired_btc_effect, got {captured.get('n_boot_btc')}"
    )


def test_portfolio_nboot_knob_default_is_2000(monkeypatch, tmp_path):
    """When MMI_PORTFOLIO_N_BOOT is unset both functions get the default 2000."""
    captured = _run_cmd_portfolio_with_nboot(monkeypatch, tmp_path, None)
    assert captured.get("n_boot_stats") == 2000, (
        f"Expected default n_boot=2000, got {captured.get('n_boot_stats')}"
    )
    assert captured.get("n_boot_btc") == 2000, (
        f"Expected default n_boot=2000 into paired_btc_effect, got {captured.get('n_boot_btc')}"
    )


@pytest.mark.parametrize("bad_value", ["", "abc", "2000.5"])
def test_portfolio_nboot_knob_invalid_string_falls_back_to_2000(monkeypatch, tmp_path, bad_value):
    """A non-integer value warns and falls back to 2000 instead of crashing the run (ValueError)."""
    captured = _run_cmd_portfolio_with_nboot(monkeypatch, tmp_path, bad_value)
    assert captured.get("n_boot_stats") == 2000, (
        f"Expected fallback n_boot=2000 for {bad_value!r}, got {captured.get('n_boot_stats')}"
    )
    assert captured.get("n_boot_btc") == 2000, (
        f"Expected fallback n_boot=2000 for {bad_value!r}, got {captured.get('n_boot_btc')}"
    )


@pytest.mark.parametrize("bad_value", ["0", "-5"])
def test_portfolio_nboot_knob_non_positive_falls_back_to_2000(monkeypatch, tmp_path, bad_value):
    """A non-positive value would degenerate the bootstrap; warn and fall back to 2000."""
    captured = _run_cmd_portfolio_with_nboot(monkeypatch, tmp_path, bad_value)
    assert captured.get("n_boot_stats") == 2000, (
        f"Expected fallback n_boot=2000 for {bad_value!r}, got {captured.get('n_boot_stats')}"
    )
    assert captured.get("n_boot_btc") == 2000, (
        f"Expected fallback n_boot=2000 for {bad_value!r}, got {captured.get('n_boot_btc')}"
    )


# ---------------------------------------------------------------------------
# D7 — manifest, atomicity, preservation invariant
# ---------------------------------------------------------------------------


def _marts_db_full(path) -> None:
    """Seed a DB with FULL marts (including portfolio + market_brief tables)."""
    con = duckdb.connect(str(path))
    con.execute("create schema if not exists marts")
    con.execute(
        "create table marts.dim_asset as "
        "select * from (values ('SPY', 'equities'), ('BTC', 'crypto')) t(symbol, asset_class)"
    )
    con.execute(
        "create table marts.fct_portfolio_returns as select * from (values "
        "('ex_btc_2002', 'equal_weight', DATE '2020-01-01', 0.01)) "
        "t(window_id, strategy, date, daily_return)"
    )
    con.execute(
        "create table marts.market_brief as select * from (values "
        "(TIMESTAMPTZ '2020-01-01 00:00:00+00', 'offline', 'brief text')) "
        "t(created_at, engine, brief)"
    )
    con.close()


def _marts_db_daily(path) -> None:
    """Seed a DB with only the cheap daily marts (NO portfolio or market_brief tables)."""
    con = duckdb.connect(str(path))
    con.execute("create schema if not exists marts")
    con.execute(
        "create table marts.dim_asset as "
        "select * from (values ('SPY', 'equities'), ('BTC', 'crypto'), ('GLD', 'commodities')) "
        "t(symbol, asset_class)"
    )
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        "('SPY', DATE '2020-01-02', 0.002, 'equities')) "
        "t(symbol, date, daily_return, asset_class)"
    )
    con.close()


def test_snapshot_writes_manifest_json(monkeypatch, tmp_path):
    """cmd_snapshot writes _manifest.json with exported table names + row counts."""
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))

    assert cli.cmd_snapshot(argparse.Namespace()) == 0

    manifest_path = out / "_manifest.json"
    assert manifest_path.exists(), "_manifest.json must be written"
    manifest = json.loads(manifest_path.read_text())

    # All exported tables appear in the manifest.
    assert set(manifest["tables"].keys()) == {"dim_asset", "fct_portfolio_returns"}
    assert manifest["tables"]["dim_asset"]["rows"] == 2
    assert manifest["tables"]["fct_portfolio_returns"]["rows"] == 1
    # generated_at must be a parseable ISO timestamp with timezone info.
    from datetime import datetime

    dt = datetime.fromisoformat(manifest["generated_at"])
    assert dt.tzinfo is not None, "generated_at must be timezone-aware"


def test_snapshot_atomicity_mid_export_failure_leaves_good_parquet_intact(monkeypatch, tmp_path):
    """A failure during a second table's export must NOT corrupt the first table's parquet.

    We pre-seed a known-good dim_asset.parquet in the output dir, then cause DuckDB's COPY
    to fail for fct_portfolio_returns.  After the exception, dim_asset.parquet must be
    byte-identical to the pre-seeded file (atomicity preserved it).
    """
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    out.mkdir()
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)

    # Pre-seed the known-good dim_asset.parquet.
    with duckdb.connect(str(db), read_only=True) as con_pre:
        con_pre.execute(f"copy marts.dim_asset to '{out / 'dim_asset.parquet'}' (format parquet)")

    good_bytes = (out / "dim_asset.parquet").read_bytes()

    # Patch connect so we get the real DB, but intercept the COPY for portfolio_returns
    # by wrapping DuckDB's execute call.
    original_connect = duckdb.connect

    class _FailingCon:
        """Wraps a real DuckDB connection; raises on the portfolio_returns COPY."""

        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *a, **k):
            if "fct_portfolio_returns" in sql and "copy" in sql.lower():
                raise RuntimeError("simulated mid-export failure")
            return self._inner.execute(sql, *a, **k)

        def fetchall(self):
            return self._inner.fetchall()

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    def _patched_connect(path=None, read_only=False, **k):
        inner = original_connect(str(db), read_only=read_only)
        return _FailingCon(inner)

    monkeypatch.setattr(cli, "connect", _patched_connect)

    with pytest.raises(RuntimeError, match="simulated mid-export failure"):
        cli.cmd_snapshot(argparse.Namespace())

    # The pre-seeded dim_asset.parquet must be byte-identical (not replaced by a partial write).
    assert (out / "dim_asset.parquet").read_bytes() == good_bytes, (
        "dim_asset.parquet was corrupted by the failed mid-export"
    )
    # No stray temp files should remain.
    assert not list(out.glob("*.tmp")), "temp files must be cleaned up on failure"


def test_snapshot_manifest_write_is_atomic(monkeypatch, tmp_path):
    """The manifest is written atomically too, not just the parquets.

    A failure during the manifest's atomic rename must leave any pre-existing
    _manifest.json byte-identical (never truncated/clobbered) and clean up the
    temp file — mirroring the per-parquet atomicity guarantee.
    """
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    out.mkdir()
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))

    # Pre-seed a known-good manifest; it must survive a failed re-write.
    prior = out / "_manifest.json"
    prior_text = '{"tables": {"OLD": {"rows": 7}}, "generated_at": "2020-01-01T00:00:00+00:00"}'
    prior.write_text(prior_text)

    real_replace = os.replace

    def failing_replace(src, dst):
        # Let the parquet renames succeed; fail only the manifest's atomic rename.
        if str(dst).endswith("_manifest.json"):
            raise RuntimeError("simulated manifest rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(RuntimeError, match="simulated manifest rename failure"):
        cli.cmd_snapshot(argparse.Namespace())

    # The pre-existing manifest must be byte-identical (atomic rename never clobbered it).
    assert prior.read_text() == prior_text, (
        "pre-existing _manifest.json must be preserved on failure"
    )
    # The manifest temp file must be cleaned up.
    assert not list(out.glob("_manifest_*.json.tmp")), (
        "manifest temp files must be cleaned up on failure"
    )


def test_snapshot_preservation_invariant(monkeypatch, tmp_path):
    """Daily-cron preservation invariant:

    1.  Snapshot a FULL marts set (dim_asset + fct_portfolio_returns + market_brief).
    2.  Drop portfolio + market_brief tables from the DB (simulating a daily-cron-only run).
    3.  Re-run mmi snapshot into the SAME output directory.
    4.  Assert:
        - fct_portfolio_returns.parquet and market_brief.parquet are BYTE-IDENTICAL (preserved).
        - dim_asset.parquet content reflects the new run (NOT identical — we change the row count).
    """
    db = tmp_path / "m.duckdb"
    _marts_db_full(db)
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))

    # STEP 1: Full snapshot.
    assert cli.cmd_snapshot(argparse.Namespace()) == 0
    portfolio_bytes = (out / "fct_portfolio_returns.parquet").read_bytes()
    brief_bytes = (out / "market_brief.parquet").read_bytes()

    # STEP 2: Drop portfolio + market_brief from the DB, add a new row to dim_asset.
    with duckdb.connect(str(db)) as con:
        con.execute("drop table marts.fct_portfolio_returns")
        con.execute("drop table marts.market_brief")
        # Modify dim_asset so its parquet will differ.
        con.execute("insert into marts.dim_asset values ('TLT', 'bonds')")

    # STEP 3: Re-run snapshot (daily-cron simulation: only dim_asset in DB now).
    assert cli.cmd_snapshot(argparse.Namespace()) == 0

    # STEP 4: Preservation + refresh assertions.
    assert (out / "fct_portfolio_returns.parquet").read_bytes() == portfolio_bytes, (
        "fct_portfolio_returns.parquet must be preserved (not overwritten) on daily re-run"
    )
    assert (out / "market_brief.parquet").read_bytes() == brief_bytes, (
        "market_brief.parquet must be preserved (not overwritten) on daily re-run"
    )
    # dim_asset was in the DB for both runs; its parquet must reflect the new row.
    with duckdb.connect() as rt:
        n = rt.execute(f"select count(*) from '{out / 'dim_asset.parquet'}'").fetchone()[0]
    assert n == 3, f"dim_asset.parquet should have 3 rows after re-run, got {n}"

    # Manifest must list only the tables exported in this second run.
    manifest = json.loads((out / "_manifest.json").read_text())
    assert "dim_asset" in manifest["tables"]
    # Portfolio and brief tables were NOT in the DB this run, so not in the manifest.
    assert "fct_portfolio_returns" not in manifest["tables"]
    assert "market_brief" not in manifest["tables"]


# ---------------------------------------------------------------------------
# D6 — fail-loud size cap (MMI_SNAPSHOT_MAX_BYTES)
# ---------------------------------------------------------------------------


def test_snapshot_size_cap_normal_passes(monkeypatch, tmp_path):
    """A normal (small) snapshot stays under the default cap → exit 0."""
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))
    # Ensure the env var is absent so the default 2_000_000 cap applies.
    monkeypatch.delenv("MMI_SNAPSHOT_MAX_BYTES", raising=False)

    assert cli.cmd_snapshot(argparse.Namespace()) == 0


def test_snapshot_size_cap_exceeded_exits_nonzero(monkeypatch, tmp_path, capsys):
    """Monkeypatching MMI_SNAPSHOT_MAX_BYTES to 1 forces the cap to trigger → exit 1."""
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))
    monkeypatch.setenv("MMI_SNAPSHOT_MAX_BYTES", "1")

    result = cli.cmd_snapshot(argparse.Namespace())

    assert result == 1, "cmd_snapshot must exit non-zero when the size cap is exceeded"

    # The error message must name both the total and the cap.
    captured = capsys.readouterr()
    assert "cap" in captured.err.lower() or "exceed" in captured.err.lower(), (
        "stderr must mention the cap or 'exceed'"
    )
    # Parquets must still be present (we export first, cap check is after).
    assert any(out.glob("*.parquet")), "parquets must be written before the cap check"


def test_snapshot_size_cap_message_names_total_and_cap(monkeypatch, tmp_path, capsys):
    """The cap-exceeded stderr message must include the numeric total and the cap value."""
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))
    cap = 1
    monkeypatch.setenv("MMI_SNAPSHOT_MAX_BYTES", str(cap))

    cli.cmd_snapshot(argparse.Namespace())
    captured = capsys.readouterr()

    assert str(cap) in captured.err, "stderr must contain the cap value"
    # The total bytes should appear somewhere in the message (may be formatted with commas).
    total = sum(p.stat().st_size for p in out.glob("*.parquet"))
    # Strip commas for comparison since the message formats with {:,}.
    assert str(total).replace(",", "") in captured.err.replace(",", ""), (
        "stderr must contain the total bytes"
    )


def test_snapshot_size_cap_whole_schema_exported_before_check(monkeypatch, tmp_path):
    """Even when the cap is exceeded ALL marts are exported (cap check is post-export)."""
    db = tmp_path / "m.duckdb"
    _marts_db(db)  # dim_asset + fct_portfolio_returns
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))
    monkeypatch.setenv("MMI_SNAPSHOT_MAX_BYTES", "1")

    result = cli.cmd_snapshot(argparse.Namespace())

    assert result == 1
    # Both tables must be present — the export is whole-schema, not trimmed.
    exported = {p.name for p in out.glob("*.parquet")}
    assert "dim_asset.parquet" in exported
    assert "fct_portfolio_returns.parquet" in exported


def test_snapshot_manifest_still_written_when_cap_exceeded(monkeypatch, tmp_path):
    """The manifest is written before the size-cap check, so it exists even on exit 1."""
    db = tmp_path / "m.duckdb"
    _marts_db(db)
    out = tmp_path / "public"
    monkeypatch.setattr(settings_mod.settings, "snapshot_dir", out)
    monkeypatch.setattr(cli, "connect", _connect_to(db))
    monkeypatch.setenv("MMI_SNAPSHOT_MAX_BYTES", "1")

    result = cli.cmd_snapshot(argparse.Namespace())

    assert result == 1
    assert (out / "_manifest.json").exists(), "_manifest.json must be written even when cap fails"
