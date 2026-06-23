"""Command-line entry point: ``mmi <command>`` (also used by the Makefile)."""

from __future__ import annotations

import argparse
import sys
import traceback

from mmi.utils.db import connect
from mmi.utils.logging import get_logger
from mmi.utils.redact import redact

log = get_logger("cli")


def cmd_seed(_: argparse.Namespace) -> int:
    """Seed deterministic sample data + build fallback marts."""
    from mmi import sampledata, transform_fallback

    with connect() as con:
        sampledata.seed(con)
        transform_fallback.build_marts(con)
    log.info("seed complete")
    return 0


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


def cmd_portfolio(_: argparse.Namespace) -> int:
    """Backtest the strategies, land returns + bootstrap-CI stats in raw.portfolio_*."""
    from mmi.ingestion import DuckDBLoader
    from mmi.portfolio.compute import (
        compute_attribution,
        compute_ml_mu_panel,
        compute_portfolio_returns,
    )
    from mmi.portfolio.stats import bootstrap_strategy_stats

    with connect() as con:
        loader = DuckDBLoader(con)
        # Exclude crypto for now: BTC is ingested into fct_asset_daily (data path) but its later
        # inception would inject staggered NaNs into this single-window panel. BTC enters the
        # backtest only once the multi-window machinery lands (Phase D5/D6), keeping this slice's
        # backtest output byte-identical to before.
        asset_daily = con.execute(
            "select symbol, date, daily_return from marts.fct_asset_daily "
            "where asset_class <> 'crypto'"
        ).df()
        # Build the ML forecast + gate ONCE, then reuse for returns + attribution (it is heavy).
        ml_mu_panel, ml_gate = compute_ml_mu_panel(asset_daily)
        results = compute_portfolio_returns(asset_daily, ml_mu_panel=ml_mu_panel)
        rows = loader.upsert("raw.portfolio_returns", results, ["strategy", "date"])
        # Honest uncertainty: stationary block-bootstrap Sharpe CIs + pairwise distinguishability.
        per_strategy, pairs = bootstrap_strategy_stats(results)
        loader.upsert("raw.portfolio_strategy_stats", per_strategy, ["strategy"])
        loader.upsert("raw.portfolio_strategy_pairs", pairs, ["strategy_a", "strategy_b"])
        # Per-asset return + risk attribution (reconciles to each strategy's gross return).
        attribution = compute_attribution(asset_daily, ml_mu_panel=ml_mu_panel)
        loader.upsert("raw.portfolio_attribution", attribution, ["strategy", "symbol"])
        # The ML gate (forecast skill + the weight it earns) makes "mvo_ml ≈ mvo_histmean" legible:
        # a low forecast_weight means the forecast showed no out-of-sample edge over the prior.
        if not ml_gate.empty:
            loader.upsert("raw.portfolio_ml_gate", ml_gate, ["date"])
    log.info(
        "portfolio: %s rows / %s strategies; %s bootstrap rows, %s pairs, %s attribution rows",
        rows,
        results["strategy"].nunique(),
        len(per_strategy),
        len(pairs),
        len(attribution),
    )
    return 0


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
