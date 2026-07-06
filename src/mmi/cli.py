"""Command-line entry point: ``mmi <command>`` (also used by the Makefile)."""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

from mmi.utils.db import connect
from mmi.utils.logging import get_logger
from mmi.utils.redact import redact

log = get_logger("cli")


def cmd_seed(_: argparse.Namespace) -> int:
    """Seed deterministic sample data + build fallback marts + offline brief.

    After building the marts, generates the deterministic offline brief (LLM keys are
    temporarily cleared so the seed step is network-free and reproducible — no API quota
    consumed, no credentials required).  This seeds ``marts.market_brief`` so the AI tab
    is never empty after a fresh ``mmi seed`` run.
    """
    from mmi import sampledata, transform_fallback
    from mmi.ai.narrative import generate_brief

    with connect() as con:
        sampledata.seed(con)
        transform_fallback.build_marts(con)
        # Force the offline-template path: clear the LLM key on the *module-level* settings
        # object (not the pydantic class) so generate_brief -> llm.available() returns False
        # without touching env vars or spawning any network request.
        _saved_key = _clear_llm_keys()
        try:
            generate_brief(con)
        except Exception as exc:  # noqa: BLE001 - brief is best-effort; seed itself succeeded
            log.warning("seed: brief generation failed (non-fatal): %s", redact(str(exc)))
        finally:
            _restore_llm_keys(_saved_key)
    log.info("seed complete")
    return 0


def _clear_llm_keys() -> dict:
    """Blank every provider key on the settings singleton; return originals for restore."""
    from mmi.settings import settings as _s

    saved = {
        "gemini_api_key": _s.gemini_api_key,
        "groq_api_key": _s.groq_api_key,
        "anthropic_api_key": _s.anthropic_api_key,
    }
    # pydantic-settings models are normally immutable; bypass via object.__setattr__.
    for attr in saved:
        object.__setattr__(_s, attr, "")
    return saved


def _restore_llm_keys(saved: dict) -> None:
    """Restore the original provider keys after the seed brief."""
    from mmi.settings import settings as _s

    for attr, val in saved.items():
        object.__setattr__(_s, attr, val)


def cmd_ingest(_: argparse.Namespace) -> int:
    """Run every extractor against the live free APIs.

    Parallelizes API fetches (network I/O bound), then loads sequentially
    (DuckDB doesn't support concurrent writes).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from mmi.ingestion import EXTRACTORS, DuckDBLoader

    required_failures = 0
    optional_failures = 0
    with connect() as con:
        loader = DuckDBLoader(con)

        # Phase 1: Parallel fetch (network I/O bound)
        fetch_results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for cls in EXTRACTORS:
                extractor = cls(loader)
                futures[executor.submit(_fetch_extractor, extractor)] = extractor

            for future in as_completed(futures):
                extractor = futures[future]
                try:
                    df = future.result()
                    fetch_results[extractor] = df
                except Exception as exc:
                    if getattr(extractor, "required", True):
                        required_failures += 1
                        log.error("REQUIRED source %s fetch failed: %s", extractor.source, str(exc)[:100])
                    else:
                        optional_failures += 1
                        log.warning("optional source %s fetch failed: %s", extractor.source, str(exc)[:100])

        # Phase 2: Sequential load (DuckDB writes)
        for extractor, df in fetch_results.items():
            if df is None or df.empty:
                if getattr(extractor, "required", True):
                    required_failures += 1
                    log.error("REQUIRED source %s returned no data", extractor.source)
                else:
                    optional_failures += 1
                    log.warning("optional source %s returned no data", extractor.source)
                continue
            try:
                run_id = extractor.loader.start_run(extractor.source)
                validated = extractor.validate(df)
                rows = extractor.loader.upsert(extractor.table, validated, extractor.keys)
                extractor.loader.finish_run(run_id, rows, "success")
                log.info("%s: %s rows", extractor.source, rows)
            except Exception as exc:
                if getattr(extractor, "required", True):
                    required_failures += 1
                    log.error("REQUIRED source %s load failed: %s", extractor.source, str(exc)[:100])
                else:
                    optional_failures += 1
                    log.warning("optional source %s load failed: %s", extractor.source, str(exc)[:100])

    if optional_failures:
        log.warning("%d optional source(s) failed; run still successful", optional_failures)
    return 1 if required_failures else 0


def _fetch_extractor(extractor):
    """Fetch data from an extractor (network I/O bound — safe to parallelize)."""
    from mmi.utils.redact import redact
    reason = extractor.skip_reason()
    if reason:
        raise RuntimeError(f"skipped: {reason}")
    start_after = None
    if extractor.watermark_col:
        wm = extractor.loader.watermark(extractor.table, extractor.watermark_col)
        if wm:
            start_after = wm
    return extractor.fetch(start_after=start_after)


def cmd_build(_: argparse.Namespace) -> int:
    """Build marts from raw using the SQL fallback (use dbt in production)."""
    from mmi import transform_fallback

    with connect() as con:
        transform_fallback.build_marts(con)
    return 0


def cmd_ml(_: argparse.Namespace) -> int:
    """Train + score forecast and regime models, persisting metrics."""
    from mmi.ml.pipeline import run_ml

    with connect() as con:
        metrics = run_ml(con)
    log.info("ml metrics: %s", metrics)
    return 0


def cmd_ai(_: argparse.Namespace) -> int:
    """Generate the GenAI market brief."""
    from mmi.ai.narrative import generate_brief

    with connect(read_only=False) as con:
        brief = generate_brief(con)
    print("\n" + brief + "\n")
    return 0


def _total_parquet_bytes(out_dir: Path) -> int:
    """Sum the on-disk size of every ``*.parquet`` in ``out_dir`` for the size-cap check.

    A parquet could, in theory, vanish or become unreadable between the ``glob()`` and the
    ``stat()`` — concurrent deletion (FileNotFoundError), a locked file or a permission change
    (OSError).  cmd_snapshot just wrote these files in a single process, so this is a purely
    theoretical TOCTOU.  If it DOES happen we must FAIL LOUD, not silently skip the file: an
    unmeasured parquet would UNDER-count the total and could let an over-cap snapshot slip past
    the cap undetected — defeating the whole point of the fail-loud cap.  So we surface the error
    (the caller turns it into a clean non-zero exit) rather than swallowing it.
    """
    total = 0
    for p in out_dir.glob("*.parquet"):
        try:
            total += p.stat().st_size
        except OSError as exc:
            raise OSError(
                f"cannot stat {p} for the snapshot size-cap tally ({exc}); "
                "refusing to certify snapshot size"
            ) from exc
    return total


def cmd_snapshot(_: argparse.Namespace) -> int:
    """Export every table in the marts schema to Parquet for the public demo.

    The public dashboard reads this static, secret-free snapshot (no hosted DB, no MotherDuck
    token). Exporting the WHOLE marts schema means any new mart is included automatically — no
    hand-maintained table list to drift.

    Atomicity: each parquet is written to a temp file then renamed, so a mid-export failure
    cannot leave a half-written parquet in place of a previously-good one.

    Preservation: if a parquet already exists in the output directory and is NOT in the current
    marts schema (e.g. portfolio/market_brief tables absent from a daily-cron-only run), the
    existing file is left byte-identical.  Only tables present in the DB are exported.

    Manifest: after a successful export, writes data/public/_manifest.json with the list of
    exported tables, per-table row counts, and a generated_at timestamp.
    """
    import json
    import os
    import tempfile
    from datetime import datetime, timezone

    from mmi.settings import settings

    out_dir = settings.snapshot_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with connect(read_only=True) as con:
        tables = [
            row[0]
            for row in con.execute(
                "select table_name from information_schema.tables "
                "where table_schema = 'marts' order by table_name"
            ).fetchall()
        ]
        if not tables:
            log.warning("snapshot: no marts tables to export — run the pipeline first")
            return 0

        manifest: dict = {"tables": {}, "generated_at": ""}
        for table in tables:
            dest = out_dir / f"{table}.parquet"
            # Write to a temp file in the same directory, then atomically rename.
            # Same-directory temp ensures the rename is on the same filesystem.
            fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=f"_{table}_", suffix=".parquet.tmp")
            try:
                os.close(fd)  # DuckDB opens by path; release the fd first.
                con.execute(f"copy marts.\"{table}\" to '{tmp_path}' (format parquet)")
                os.replace(tmp_path, dest)  # atomic on POSIX; on Windows may raise on open handles
            except Exception:
                # Clean up the temp file; leave any pre-existing dest untouched (preservation).
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

            # Collect row count for manifest.
            row = con.execute(f'select count(*) from marts."{table}"').fetchone()
            manifest["tables"][table] = {"rows": row[0] if row else 0}

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = out_dir / "_manifest.json"
    # Write the manifest atomically too (mirror the per-parquet pattern): a crash
    # mid-write must never leave a truncated/invalid _manifest.json. Serialise to a
    # temp file in the same dir, then atomically rename; a pre-existing manifest is
    # left intact if anything fails.
    fd, tmp_manifest = tempfile.mkstemp(dir=out_dir, prefix="_manifest_", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(manifest, fh, indent=2)
        os.replace(tmp_manifest, manifest_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_manifest)
        raise

    log.info("snapshot: exported %d marts tables to %s", len(tables), out_dir)

    # --- fail-loud size cap (Contract A) ---
    # After a successful export, sum the bytes of every *.parquet in out_dir.
    # If the total exceeds MMI_SNAPSHOT_MAX_BYTES (default 12_000_000), exit non-zero with a
    # clear message so the owner notices before committing an oversized snapshot. The real
    # 24-year dataset is ~5.7 MB; 12 MB leaves years of forward-growth headroom while still
    # catching a gross runaway (a raw-data dump, an accidental .duckdb commit, a cartesian-join
    # blowup) that would be tens of MB. The remedy for a genuine future overflow is a new
    # downsampled dbt mart; do NOT trim or exclude marts from the export.
    # (``os`` is already imported at the top of this function.)
    default_max_bytes = 12_000_000
    raw_max = os.environ.get("MMI_SNAPSHOT_MAX_BYTES")
    max_bytes = default_max_bytes
    if raw_max is not None:
        try:
            parsed_max = int(raw_max)
            if parsed_max > 0:
                max_bytes = parsed_max
            else:
                log.warning(
                    "MMI_SNAPSHOT_MAX_BYTES=%d must be > 0; using default %d",
                    parsed_max,
                    default_max_bytes,
                )
        except ValueError:
            log.warning(
                "MMI_SNAPSHOT_MAX_BYTES=%r is not an integer; using default %d",
                raw_max,
                default_max_bytes,
            )

    try:
        total_bytes = _total_parquet_bytes(out_dir)
    except OSError as exc:
        # An unreadable parquet means we cannot certify the snapshot's size — fail loud
        # (a clean non-zero exit) rather than publish an unverified snapshot.
        log.error("snapshot: %s — aborting before publish", exc)
        print(f"ERROR: {exc} — aborting before publish.", file=sys.stderr)
        return 1
    if total_bytes > max_bytes:
        log.error(
            "snapshot: total parquet size %d bytes exceeds cap %d bytes — "
            "remedy is a new downsampled dbt mart, NOT trimming the export",
            total_bytes,
            max_bytes,
        )
        print(
            f"ERROR: snapshot size {total_bytes:,} bytes exceeds cap {max_bytes:,} bytes. "
            "Remedy: add a downsampled dbt mart — do NOT exclude marts from the export.",
            file=sys.stderr,
        )
        return 1

    return 0


def cmd_portfolio(_: argparse.Namespace) -> int:
    """Backtest the strategies per window, landing returns + bootstrap-CI stats in raw.portfolio_*.

    Phase D runs THREE windows (issue #7): ex_btc_2002 (the long non-crypto baseline), ex_btc_2015
    (same assets, BTC's era — the same-period control), and inc_btc_2015 (adds BTC, same period).
    Each runs as a SEPARATELY filtered panel — never one merged panel, whose dropna would collapse
    the 2002+ history to BTC's ~2015 start. The two 2015 windows share one derived BTC-inception
    floor so they are byte-identical in period; inc_btc aligns BTC to the equity calendar.

    MMI_PORTFOLIO_N_BOOT controls the bootstrap iteration count (default 2000) for BOTH
    bootstrap_strategy_stats and paired_btc_effect.  Lower values (e.g. 200) dramatically speed up
    local iteration; the env var is the single dial — never hard-code a different default here.
    It is parsed defensively: a non-integer or non-positive value warns and falls back to 2000 so a
    fat-fingered knob can never crash the whole run.
    """
    import os

    import pandas as pd

    from mmi.ingestion import DuckDBLoader
    from mmi.ingestion.loader import reset_portfolio_raw_tables
    from mmi.portfolio import compute, windows
    from mmi.portfolio.stats import bootstrap_strategy_stats, paired_btc_effect
    from mmi.settings import load_assets

    # Owner-only tuning knob: parse defensively.  A non-integer ("", "abc", "2000.5") or a
    # non-positive value (0, negative — which would degenerate the bootstrap) warns and falls back
    # to the default rather than killing the run.  Unset keeps the default-2000 behaviour unchanged.
    default_n_boot = 2000
    raw_n_boot = os.environ.get("MMI_PORTFOLIO_N_BOOT")
    n_boot = default_n_boot
    if raw_n_boot is not None:
        try:
            parsed_n_boot = int(raw_n_boot)
        except ValueError:
            log.warning(
                "MMI_PORTFOLIO_N_BOOT=%r is not an integer; falling back to %d",
                raw_n_boot,
                default_n_boot,
            )
        else:
            if parsed_n_boot > 0:
                n_boot = parsed_n_boot
            else:
                log.warning(
                    "MMI_PORTFOLIO_N_BOOT=%d must be > 0; falling back to %d",
                    parsed_n_boot,
                    default_n_boot,
                )

    def run_window(
        loader: DuckDBLoader,
        window_id: str,
        wad,
        *,
        ml_mu_override: pd.DataFrame | None = None,
    ) -> tuple[int, int, pd.DataFrame]:
        # Build the ML forecast + gate ONCE per window, then reuse for returns + attribution.
        # An ml_mu_override from the wider inc_btc_2015 universe ensures common_dates is identical
        # for both 2015 windows (required by assert_portfolio_windows_period_aligned).
        if ml_mu_override is not None:
            ml_mu_panel = ml_mu_override
            ml_gate = pd.DataFrame(columns=["date", "forecast_skill", "forecast_weight"])
        else:
            ml_mu_panel, ml_gate = compute.compute_ml_mu_panel(
                wad, window=window_id, asset_daily_full=wad,
                macro_df=macro_wide, asset_dfs=asset_dfs_macro,
            )
        results = compute.compute_portfolio_returns(
            wad, ml_mu_panel=ml_mu_panel, window=window_id, asset_daily_full=wad
        )
        n = loader.upsert("raw.portfolio_returns", results, ["window_id", "strategy", "date"])
        # Honest uncertainty: block-bootstrap Sharpe CIs. One window at a time — the bootstrap
        # pivots by date x strategy and would collide windows if handed more than one.
        per_strategy, pairs = bootstrap_strategy_stats(results, window=window_id, n_boot=n_boot)
        loader.upsert("raw.portfolio_strategy_stats", per_strategy, ["window_id", "strategy"])
        loader.upsert(
            "raw.portfolio_strategy_pairs", pairs, ["window_id", "strategy_a", "strategy_b"]
        )
        attribution = compute.compute_attribution(
            wad, ml_mu_panel=ml_mu_panel, window=window_id, asset_daily_full=wad
        )
        loader.upsert("raw.portfolio_attribution", attribution, ["window_id", "strategy", "symbol"])
        # The ML gate (forecast skill + the weight it earns) makes "mvo_ml ≈ mvo_histmean" legible.
        if not ml_gate.empty:
            loader.upsert("raw.portfolio_ml_gate", ml_gate, ["window_id", "date"])
        return n, results["strategy"].nunique(), results

    with connect() as con:
        loader = DuckDBLoader(con)
        # Recreate the wholesale-landed portfolio tables so the backtest self-heals on a stale DB
        # (schema/window changes); also clears prior windows before this full re-run.
        reset_portfolio_raw_tables(con)
        # Pull the WHOLE daily panel incl. BTC; each window filters its own universe in Python.
        asset_daily = con.execute(
            "select symbol, date, daily_return, asset_class from marts.fct_asset_daily"
        ).df()

        # Load macro data for vol_macro features
        try:
            macro_raw = con.execute(
                "select date, series_id, value from marts.fct_macro_indicator order by date"
            ).df()
            if not macro_raw.empty:
                macro_raw["date"] = pd.to_datetime(macro_raw["date"]).astype("datetime64[ns]")
                macro_wide = macro_raw.pivot_table(
                    index="date", columns="series_id", values="value", aggfunc="first"
                ).reset_index().sort_values("date")
                for col in macro_wide.columns:
                    if col != "date":
                        macro_wide[col] = macro_wide[col].ffill()
            else:
                macro_wide = None
        except Exception:
            macro_wide = None

        # Load cross-asset data for vol_macro features
        asset_dfs_macro = {}
        for sym in ["GLD", "TLT"]:
            try:
                adf = con.execute(
                    f"select date, daily_return from marts.fct_asset_daily where symbol='{sym}'"
                ).df()
                if not adf.empty:
                    adf["date"] = pd.to_datetime(adf["date"]).astype("datetime64[ns]")
                    asset_dfs_macro[sym] = adf
            except Exception:
                pass

        # BTC on the equity trading calendar defines the shared 2015 floor (its first valid return).
        btc_aligned = compute.btc_aligned_returns(asset_daily)
        valid = btc_aligned.dropna(subset=["daily_return"])
        btc_floor = valid["date"].min() if not valid.empty else None
        if btc_floor is None and load_assets().get("crypto_daily"):
            # Loud, not silent: BTC is configured but missing, so the BTC-era windows can't build.
            log.warning(
                "BTC declared in config but absent from fct_asset_daily; skipping 2015 windows"
            )

        ran: list[str] = []
        results_by_window: dict[str, pd.DataFrame] = {}

        # Pre-compute ML panels for all windows in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        window_data = {}
        for window_id in windows.WINDOWS:
            if window_id != windows.EX_BTC_2002 and btc_floor is None:
                continue
            wad = compute.window_asset_daily(
                asset_daily, window_id, btc_floor=btc_floor, btc_aligned=btc_aligned
            )
            if not wad.empty:
                window_data[window_id] = wad

        ml_panels = {}
        def _compute_ml_panel(wid, wad):
            return wid, compute.compute_ml_mu_panel(
                wad, window=wid, asset_daily_full=wad,
                macro_df=macro_wide, asset_dfs=asset_dfs_macro,
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_compute_ml_panel, wid, wad): wid
                      for wid, wad in window_data.items()}
            for future in as_completed(futures):
                wid = futures[future]
                try:
                    ml_panels[wid] = future.result()
                except Exception as exc:
                    log.warning("ML panel failed for %s: %s", wid, exc)

        # Sequential portfolio backtests (DuckDB writes)
        for window_id in window_data:
            wad = window_data[window_id]
            override = ml_panels.get(window_id)
            if override:
                ml_mu_panel, ml_gate = override
            else:
                ml_mu_panel = None
                ml_gate = pd.DataFrame()
            n, n_strategies, results = run_window(
                loader, window_id, wad,
                ml_mu_override=ml_mu_panel if ml_mu_panel is not None else None,
            )
            results_by_window[window_id] = results
            log.info("portfolio[%s]: %s rows / %s strategies", window_id, n, n_strategies)
            ran.append(window_id)

        # The BTC effect: Sharpe(inc_btc_2015) − Sharpe(ex_btc_2015) with a PAIRED cross-window
        # bootstrap CI. Valid only because the two 2015 windows are period-identical (same dates) —
        # the per-window bootstraps cannot give this CI without overstating the variance.
        if {windows.EX_BTC_2015, windows.INC_BTC_2015} <= results_by_window.keys():
            effect = paired_btc_effect(
                results_by_window[windows.EX_BTC_2015],
                results_by_window[windows.INC_BTC_2015],
                n_boot=n_boot,
            )
            if not effect.empty:
                loader.upsert("raw.portfolio_btc_effect", effect, ["strategy"])
                log.info(
                    "btc effect: %d strategies, %d distinguishable",
                    len(effect),
                    int(effect["distinguishable"].sum()),
                )
    log.info("portfolio: ran %d window(s): %s", len(ran), ", ".join(ran))
    return 0


def cmd_healthcheck(_: argparse.Namespace) -> int:
    """Probe every data source for connectivity + key presence.

    Prints a source -> ok | skip(reason) | fail(reason) table.
    Exits non-zero only if a *required* source FAILs.
    Does NOT open a DB connection.
    """
    from mmi.ingestion import EXTRACTORS
    from mmi.ingestion.healthcheck import exit_code, format_table, run_healthcheck

    results = run_healthcheck(EXTRACTORS)
    print(format_table(results))
    return exit_code(results)


def cmd_ml_gate(args: argparse.Namespace) -> int:
    """Check the HAR realized-volatility skill gate against persisted model_metrics.

    Reads ``marts.model_metrics`` via the normal DB connection and delegates ALL verdict
    logic to ``skill_verdict()`` from ``src/mmi/ml/skill_gate.py`` — the single source of
    truth for the gate (Contract E).  This command NEVER re-derives a verdict itself.

    STRICT mode (default):
      * Prints the verdict and any failure reasons.
      * Exits non-zero when the gate is NOT cleared; exits 0 when cleared.

    --warn-only mode:
      * Prints the same output but always exits 0 (useful in CI contexts where a not-yet-
        trained model should warn rather than block the pipeline).

    Absent or partial metric rows (e.g. model not yet trained) yield a ``not-cleared``
    result with an explanatory reason string — never an exception.

    NOT wired into ``make ci``: sample data has no real edge so the gate would always fail,
    which would break CI.  This command is for the owner's local pre-snapshot check only.
    """
    # Lazy import inside function: skill_gate has no module-scope ML lib imports, but we
    # follow the convention of all other cmd_* functions to avoid import-time side effects.
    from mmi.ml.skill_gate import skill_verdict

    symbol: str = args.symbol
    warn_only: bool = args.warn_only

    # ------------------------------------------------------------------
    # Read marts.model_metrics — absent/partial rows must not raise.
    # ------------------------------------------------------------------
    import pandas as pd

    try:
        with connect(read_only=True) as con:
            try:
                metrics_df: pd.DataFrame = con.execute(
                    "select model, symbol, metric, value, trained_at from marts.model_metrics"
                ).df()
            except Exception as exc:  # noqa: BLE001 - table missing or schema mismatch
                log.warning("ml-gate: could not read marts.model_metrics: %s", redact(str(exc)))
                metrics_df = pd.DataFrame(
                    columns=["model", "symbol", "metric", "value", "trained_at"]
                )
    except Exception as exc:  # noqa: BLE001 - DB connection failure
        log.warning("ml-gate: DB connection failed: %s", redact(str(exc)))
        metrics_df = pd.DataFrame(columns=["model", "symbol", "metric", "value", "trained_at"])

    # ------------------------------------------------------------------
    # Delegate to the single source of truth for the verdict.
    # ------------------------------------------------------------------
    verdict = skill_verdict(metrics_df, symbol=symbol)

    cleared: bool = verdict["cleared"]
    reasons: list[str] = verdict["reasons"]

    # ------------------------------------------------------------------
    # Print the verdict.
    # ------------------------------------------------------------------
    if cleared:
        print(f"ml-gate: CLEARED — symbol={symbol}, model=rv_har")
        print(
            f"  oos_r2={verdict['oos_r2']:.4f}  "
            f"qlike_skill_ratio={verdict['qlike_skill_ratio']:.4f}  "
            f"folds_passed={verdict['folds_passed']}/{verdict['n_folds']}  "
            f"n_obs={verdict['n_obs']}"
        )
    else:
        print(f"ml-gate: NOT CLEARED — symbol={symbol}, model=rv_har")
        for reason in reasons:
            print(f"  reason: {reason}")

    if warn_only and not cleared:
        log.warning("ml-gate: not cleared (warn-only mode — exit 0)")
        return 0

    return 0 if cleared else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmi", description="Markets & Macro Intelligence CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, help_ in [
        ("seed", cmd_seed, "Seed sample data + fallback marts"),
        ("ingest", cmd_ingest, "Pull live data from free APIs"),
        ("build", cmd_build, "Build marts from raw (SQL fallback)"),
        ("ml", cmd_ml, "Train/score ML models"),
        ("ai", cmd_ai, "Generate GenAI market brief"),
        ("portfolio", cmd_portfolio, "Backtest portfolio strategies -> raw.portfolio_returns"),
        ("snapshot", cmd_snapshot, "Export marts.* to Parquet for the public demo"),
        ("healthcheck", cmd_healthcheck, "Probe every data source for connectivity + key presence"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.set_defaults(func=fn)
    # ml-gate has extra arguments so it is registered separately.
    p_ml_gate = sub.add_parser(
        "ml-gate",
        help="Check HAR realized-vol skill gate against persisted model_metrics (not in make ci)",
    )
    p_ml_gate.set_defaults(func=cmd_ml_gate)
    p_ml_gate.add_argument(
        "--symbol",
        default="SPY",
        metavar="TICKER",
        help="Asset ticker to evaluate (default: SPY)",
    )
    p_ml_gate.add_argument(
        "--warn-only",
        action="store_true",
        default=False,
        help="Print verdict but always exit 0 (never blocks the pipeline)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception:  # noqa: BLE001 - redact any error before it reaches stderr / CI logs
        log.error("command '%s' failed:\n%s", args.command, redact(traceback.format_exc()))
        return 1


if __name__ == "__main__":
    sys.exit(main())
