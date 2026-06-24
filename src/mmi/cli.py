"""Command-line entry point: ``mmi <command>`` (also used by the Makefile)."""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    A failure in a *required* source fails the run (exit 1) so scheduled jobs cannot go green
    on a broken pipeline. *Optional* sources (e.g. unofficial endpoints like Stooq) are recorded
    in ``raw.pipeline_runs`` and surfaced as warnings, but do not fail the ingest step.

    Optional sources stay non-fatal even on a *fresh* database: ``DuckDBLoader`` pre-creates the
    raw source tables empty (``ensure_raw_tables``), so dbt builds empty marts for a not-yet-loaded
    source instead of erroring on a missing one. A *required* failure still fails the run.
    """
    from mmi.ingestion import EXTRACTORS, DuckDBLoader

    required_failures = 0
    optional_failures = 0
    with connect() as con:
        loader = DuckDBLoader(con)
        for cls in EXTRACTORS:
            extractor = cls(loader)
            try:
                rows = extractor.run()
                log.info("%s: %s rows", extractor.source, rows)
            except Exception as exc:  # noqa: BLE001 - record, classify, keep going
                if getattr(extractor, "required", True):
                    required_failures += 1
                    log.error("REQUIRED source %s failed: %s", extractor.source, redact(str(exc)))
                else:
                    optional_failures += 1
                    log.warning(
                        "optional source %s failed (continuing): %s",
                        extractor.source,
                        redact(str(exc)),
                    )
    if optional_failures:
        log.warning("%d optional source(s) failed; run still successful", optional_failures)
    return 1 if required_failures else 0


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


def cmd_snapshot(_: argparse.Namespace) -> int:
    """Export every table in the marts schema to Parquet for the public demo.

    The public dashboard reads this static, secret-free snapshot (no hosted DB, no MotherDuck
    token). Exporting the WHOLE marts schema means any new mart is included automatically — no
    hand-maintained table list to drift.
    """
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
        for table in tables:
            path = out_dir / f"{table}.parquet"
            con.execute(f"copy marts.\"{table}\" to '{path}' (format parquet)")
    log.info("snapshot: exported %d marts tables to %s", len(tables), out_dir)
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
    """
    import os

    from mmi.ingestion import DuckDBLoader
    from mmi.ingestion.loader import reset_portfolio_raw_tables
    from mmi.portfolio import compute, windows
    from mmi.portfolio.stats import bootstrap_strategy_stats, paired_btc_effect
    from mmi.settings import load_assets

    n_boot: int = int(os.environ.get("MMI_PORTFOLIO_N_BOOT", 2000))

    def run_window(loader: DuckDBLoader, window_id: str, wad) -> tuple[int, int, pd.DataFrame]:
        # Build the ML forecast + gate ONCE per window, then reuse for returns + attribution.
        ml_mu_panel, ml_gate = compute.compute_ml_mu_panel(wad, window=window_id)
        results = compute.compute_portfolio_returns(wad, ml_mu_panel=ml_mu_panel, window=window_id)
        n = loader.upsert("raw.portfolio_returns", results, ["window_id", "strategy", "date"])
        # Honest uncertainty: block-bootstrap Sharpe CIs. One window at a time — the bootstrap
        # pivots by date x strategy and would collide windows if handed more than one.
        per_strategy, pairs = bootstrap_strategy_stats(results, window=window_id, n_boot=n_boot)
        loader.upsert("raw.portfolio_strategy_stats", per_strategy, ["window_id", "strategy"])
        loader.upsert(
            "raw.portfolio_strategy_pairs", pairs, ["window_id", "strategy_a", "strategy_b"]
        )
        attribution = compute.compute_attribution(wad, ml_mu_panel=ml_mu_panel, window=window_id)
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
        for window_id in windows.WINDOWS:
            if window_id != windows.EX_BTC_2002 and btc_floor is None:
                continue  # 2015 windows need the BTC floor (warned above)
            wad = compute.window_asset_daily(
                asset_daily, window_id, btc_floor=btc_floor, btc_aligned=btc_aligned
            )
            n, n_strategies, results = run_window(loader, window_id, wad)
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
