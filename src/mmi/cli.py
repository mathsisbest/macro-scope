"""Command-line entry point: ``mmi <command>`` (also used by the Makefile)."""

from __future__ import annotations

import argparse
import sys

from mmi.utils.db import connect
from mmi.utils.logging import get_logger

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
    """Run every extractor against the live free APIs."""
    from mmi.ingestion import EXTRACTORS, DuckDBLoader

    failures = 0
    with connect() as con:
        loader = DuckDBLoader(con)
        for cls in EXTRACTORS:
            extractor = cls(loader)
            try:
                rows = extractor.run()
                log.info("%s: %s rows", extractor.source, rows)
            except Exception as exc:  # noqa: BLE001 - keep ingesting other sources
                failures += 1
                log.error("%s failed: %s", extractor.source, exc)
    return 1 if failures else 0


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
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
