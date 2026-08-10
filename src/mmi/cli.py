"""Command-line entry point: ``mmi <command>`` (also used by the Makefile)."""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from mmi.ingestion.base import Extractor

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


def _fetch_all_extractors(loader) -> dict[str, tuple[Extractor, str, Any]]:
    """Phase 0 & Phase 1 of ingestion: sequential watermarks + parallel network fetches."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from mmi.ingestion import EXTRACTORS

    start_afters: dict[str, str | None] = {}
    extractors: list[Extractor] = []
    for cls in EXTRACTORS:
        ext = cls(loader)
        extractors.append(ext)
        wm: str | None = None
        if ext.watermark_col:
            with contextlib.suppress(Exception):
                wm = loader.watermark(ext.table, ext.watermark_col)
        start_afters[ext.source] = wm

    fetch_results: dict[str, tuple[Extractor, str, Any]] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_fetch_extractor, ext, start_afters[ext.source]): ext
            for ext in extractors
        }

        for future in as_completed(futures):
            extractor = futures[future]
            try:
                res_type, payload = future.result()
                fetch_results[extractor.source] = (extractor, res_type, payload)
            except Exception as exc:
                fetch_results[extractor.source] = (extractor, "error", exc)

    return fetch_results


def _load_ingested_results(loader, fetch_results) -> tuple[int, int]:
    """Phase 2 of ingestion: sequential load & audit logging."""
    required_failures = 0
    optional_failures = 0

    for extractor, res_type, payload in fetch_results.values():
        run_id = loader.start_run(extractor.source)

        if res_type == "skip":
            reason = payload
            log.warning("%s: skipping — %s", extractor.source, reason)
            loader.finish_run(run_id, 0, "skipped", reason)
            continue

        if res_type == "error":
            err = payload
            msg = redact(str(err))
            loader.finish_run(run_id, 0, "failed", msg)
            if getattr(extractor, "required", True):
                required_failures += 1
                log.error("REQUIRED source %s fetch failed: %s", extractor.source, msg[:100])
            else:
                optional_failures += 1
                log.warning("optional source %s fetch failed: %s", extractor.source, msg[:100])
            continue

        df = payload
        if df is None or df.empty:
            loader.finish_run(run_id, 0, "success")
            log.info("%s: 0 rows", extractor.source)
            continue

        try:
            validated = extractor.validate(df)
            rows = loader.upsert(extractor.table, validated, extractor.keys)
            loader.finish_run(run_id, rows, "success")
            log.info("%s: %s rows", extractor.source, rows)
        except Exception as exc:
            msg = redact(str(exc))
            loader.finish_run(run_id, 0, "failed", msg)
            if getattr(extractor, "required", True):
                required_failures += 1
                log.error("REQUIRED source %s load failed: %s", extractor.source, msg[:100])
            else:
                optional_failures += 1
                log.warning("optional source %s load failed: %s", extractor.source, msg[:100])

    return required_failures, optional_failures


def cmd_ingest(_: argparse.Namespace) -> int:
    """Run every extractor against the live free APIs.

    Parallelizes API fetches (network I/O bound), then loads sequentially
    (DuckDB doesn't support concurrent writes).
    """
    from mmi.ingestion import DuckDBLoader

    with connect() as con:
        loader = DuckDBLoader(con)
        fetch_results = _fetch_all_extractors(loader)
        req_fails, opt_fails = _load_ingested_results(loader, fetch_results)

    if opt_fails:
        log.warning("%d optional source(s) failed; run still successful", opt_fails)
    return 1 if req_fails else 0


def _fetch_extractor(extractor: Extractor, start_after: str | None = None) -> tuple[str, Any]:
    """Fetch data from an extractor (network I/O bound — safe to parallelize).

    ``start_after`` is computed **sequentially** before the parallel phase so the
    sequential Phase 0 sets a consistent baseline.
    """
    reason = extractor.skip_reason()
    if reason:
        return ("skip", reason)
    df = extractor.fetch(start_after=start_after)
    return ("ok", df)


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
    """Sum the sizes of all *.parquet files in out_dir, following symlinks.

    Raises OSError on any stat failure so the caller can fail-loud rather than publish
    an uncertified snapshot.
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


def _export_marts_tables_to_parquet(con, out_dir: Path) -> tuple[list[str], dict]:
    """Export every marts schema table to Parquet and return table list + manifest data."""
    import os
    import tempfile

    from mmi.utils.atomic import atomic_replace

    tables = [
        row[0]
        for row in con.execute(
            "select table_name from information_schema.tables "
            "where table_schema = 'marts' order by table_name"
        ).fetchall()
    ]
    if not tables:
        return [], {}

    manifest: dict = {"tables": {}, "generated_at": ""}
    for table in tables:
        dest = out_dir / f"{table}.parquet"
        fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=f"_{table}_", suffix=".parquet.tmp")
        try:
            os.close(fd)
            con.execute(f"copy marts.\"{table}\" to '{tmp_path}' (format parquet)")
            atomic_replace(tmp_path, dest)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        row = con.execute(f'select count(*) from marts."{table}"').fetchone()
        manifest["tables"][table] = {"rows": row[0] if row else 0}

    return tables, manifest


def _write_snapshot_manifest(out_dir: Path, manifest: dict) -> None:
    """Atomically write data/public/_manifest.json."""
    import json
    from datetime import datetime, timezone

    from mmi.utils.atomic import atomic_write

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write(out_dir / "_manifest.json", json.dumps(manifest, indent=2))


def cmd_snapshot(_: argparse.Namespace) -> int:
    """Export every table in the marts schema to Parquet for the public demo."""
    import sys

    from mmi.settings import Settings, settings

    out_dir = settings.snapshot_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with connect(read_only=True) as con:
        tables, manifest = _export_marts_tables_to_parquet(con, out_dir)
        if not tables:
            log.warning("snapshot: no marts tables to export — run the pipeline first")
            return 0
        _write_snapshot_manifest(out_dir, manifest)

    log.info("snapshot: exported %d marts tables to %s", len(tables), out_dir)

    max_bytes = Settings().snapshot_max_bytes

    try:
        total_bytes = _total_parquet_bytes(out_dir)
    except OSError as exc:
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


def _load_portfolio_macro_context(con) -> tuple[Any, dict]:
    """Load macro indicators and cross-asset DataFrames for portfolio feature engineering."""
    import pandas as pd

    try:
        macro_raw = con.execute(
            "select date, series_id, value from marts.fct_macro_indicator order by date"
        ).df()
        if not macro_raw.empty:
            macro_raw["date"] = pd.to_datetime(macro_raw["date"]).astype("datetime64[ns]")
            macro_wide = (
                macro_raw.pivot_table(
                    index="date", columns="series_id", values="value", aggfunc="first"
                )
                .reset_index()
                .sort_values("date")
            )
            for col in macro_wide.columns:
                if col != "date":
                    macro_wide[col] = macro_wide[col].ffill()
        else:
            macro_wide = None
    except Exception:
        macro_wide = None

    asset_dfs_macro = {}
    for sym in ["GLD", "TLT"]:
        try:
            adf = con.execute(
                "select date, daily_return from marts.fct_asset_daily where symbol = ?",
                [sym],
            ).df()
            if not adf.empty:
                adf["date"] = pd.to_datetime(adf["date"]).astype("datetime64[ns]")
                asset_dfs_macro[sym] = adf
        except Exception:
            pass

    return macro_wide, asset_dfs_macro


def _run_single_portfolio_window(
    loader,
    window_id: str,
    wad,
    macro_wide,
    asset_dfs_macro,
    n_boot: int,
    *,
    ml_mu_override=None,
):
    """Run a single portfolio window backtest, computing returns, stats, and attribution."""
    import pandas as pd

    from mmi.portfolio import compute
    from mmi.portfolio.stats import bootstrap_strategy_stats

    if ml_mu_override is not None:
        ml_mu_panel = ml_mu_override
        ml_gate = pd.DataFrame(columns=["date", "forecast_skill", "forecast_weight"])
    else:
        ml_mu_panel, ml_gate = compute.compute_ml_mu_panel(
            wad,
            window=window_id,
            asset_daily_full=wad,
            macro_df=macro_wide,
            asset_dfs=asset_dfs_macro,
        )
    results = compute.compute_portfolio_returns(
        wad, ml_mu_panel=ml_mu_panel, window=window_id, asset_daily_full=wad
    )
    n = loader.upsert("raw.portfolio_returns", results, ["window_id", "strategy", "date"])
    per_strategy, pairs = bootstrap_strategy_stats(results, window=window_id, n_boot=n_boot)
    loader.upsert("raw.portfolio_strategy_stats", per_strategy, ["window_id", "strategy"])
    loader.upsert("raw.portfolio_strategy_pairs", pairs, ["window_id", "strategy_a", "strategy_b"])
    attribution = compute.compute_attribution(
        wad, ml_mu_panel=ml_mu_panel, window=window_id, asset_daily_full=wad
    )
    loader.upsert("raw.portfolio_attribution", attribution, ["window_id", "strategy", "symbol"])
    if not ml_gate.empty:
        loader.upsert("raw.portfolio_ml_gate", ml_gate, ["window_id", "date"])
    return n, results["strategy"].nunique(), results


def cmd_portfolio(_: argparse.Namespace) -> int:
    """Backtest the strategies per window, landing returns in raw.portfolio_*."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from mmi.ingestion import DuckDBLoader
    from mmi.ingestion.loader import reset_portfolio_raw_tables
    from mmi.portfolio import compute, windows
    from mmi.portfolio.stats import paired_btc_effect
    from mmi.settings import Settings, load_assets

    n_boot = Settings().portfolio_n_boot

    with connect() as con:
        loader = DuckDBLoader(con)
        reset_portfolio_raw_tables(con)
        asset_daily = con.execute(
            "select symbol, date, open, high, low, close, "
            "daily_return, asset_class from marts.fct_asset_daily"
        ).df()

        macro_wide, asset_dfs_macro = _load_portfolio_macro_context(con)

        btc_aligned = compute.btc_aligned_returns(asset_daily)
        valid = btc_aligned.dropna(subset=["daily_return"])
        btc_floor = valid["date"].min() if not valid.empty else None
        if btc_floor is None and load_assets().get("crypto_daily"):
            log.warning(
                "BTC declared in config but absent from fct_asset_daily; skipping 2015 windows"
            )

        ran: list[str] = []
        results_by_window: dict[str, pd.DataFrame] = {}

        ml_mu_2015: pd.DataFrame | None = None
        if btc_floor is not None:
            wad_wide = compute.window_asset_daily(
                asset_daily,
                windows.INC_BTC_2015,
                btc_floor=btc_floor,
                btc_aligned=btc_aligned,
            )
            if not wad_wide.empty:
                ml_mu_2015, _ = compute.compute_ml_mu_panel(
                    wad_wide,
                    window=windows.INC_BTC_2015,
                    asset_daily_full=wad_wide,
                    macro_df=macro_wide,
                    asset_dfs=asset_dfs_macro,
                )

        window_data = {}
        for window_id in windows.WINDOWS:
            if window_id != windows.EX_BTC_2002 and btc_floor is None:
                continue
            wad = compute.window_asset_daily(
                asset_daily, window_id, btc_floor=btc_floor, btc_aligned=btc_aligned
            )
            if not wad.empty:
                window_data[window_id] = wad

        ml_panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

        def _compute_ml_panel(wid, wad):
            return compute.compute_ml_mu_panel(
                wad,
                window=wid,
                asset_daily_full=wad,
                macro_df=macro_wide,
                asset_dfs=asset_dfs_macro,
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_compute_ml_panel, wid, wad): wid
                for wid, wad in window_data.items()
            }
            for future in as_completed(futures):
                wid = futures[future]
                try:
                    ml_panels[wid] = future.result()
                except Exception as exc:
                    log.warning("ML panel failed for %s: %s", wid, exc)

        for window_id in window_data:
            wad = window_data[window_id]
            if window_id in (windows.EX_BTC_2015, windows.INC_BTC_2015) and ml_mu_2015 is not None:
                override = ml_mu_2015
            else:
                result = ml_panels.get(window_id)
                override = result[0] if result else None
            n, n_strategies, results = _run_single_portfolio_window(
                loader,
                window_id,
                wad,
                macro_wide,
                asset_dfs_macro,
                n_boot,
                ml_mu_override=override,
            )
            results_by_window[window_id] = results
            log.info("portfolio[%s]: %s rows / %s strategies", window_id, n, n_strategies)
            ran.append(window_id)

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
