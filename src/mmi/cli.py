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

    Caveat: "optional" only protects the *ingest step*. If an optional source is the sole
    producer of a raw table that dbt requires (Stooq -> raw.asset_prices), the downstream
    ``dbt build`` still fails on a *fresh* database where that table has never loaded; on a
    populated database a transient failure is tolerated (prior data remains). First-run
    robustness is tracked as a follow-up.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmi", description="Markets & Macro Intelligence CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, help_ in [
        ("seed", cmd_seed, "Seed sample data + fallback marts"),
        ("ingest", cmd_ingest, "Pull live data from free APIs"),
        ("build", cmd_build, "Build marts from raw (SQL fallback)"),
        ("ml", cmd_ml, "Train/score ML models"),
        ("ai", cmd_ai, "Generate GenAI market brief"),
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
