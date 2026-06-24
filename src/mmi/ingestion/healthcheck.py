"""Healthcheck: probe every registered source for key presence + connectivity.

Usage via CLI: ``mmi healthcheck``

Prints a source -> ok | skip(reason) | fail(reason) table.
Exits non-zero only if a *required* source FAILs.
skip-before-probe ordering is enforced: if skip_reason() is set the probe is never called.
Redaction is applied at the classify_source() boundary — the single chokepoint — so no raw
exception string (which may embed API keys from URL query params) reaches the CLI or any log.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from mmi.ingestion.base import Extractor
from mmi.utils.redact import redact

HealthStatus = Literal["ok", "skip", "fail"]


@dataclass(frozen=True)
class ProbeResult:
    source: str  # extractor.source, e.g. "fred"
    status: HealthStatus
    required: bool  # extractor.required (drives exit code)
    # redacted reason; "" for plain ok; skip=skip_reason(); fail=redacted exc str
    detail: str = field(default="")


def classify_source(extractor: Extractor) -> ProbeResult:
    """Determine health status for one extractor instance.

    Order: skip_reason() FIRST (non-None -> "skip", probe NOT called).
    Then probe(): no exception -> "ok"; Exception -> "fail" with detail=redact(str(exc)).
    Redaction happens here — this is the single chokepoint before strings leave this module.
    """
    reason = extractor.skip_reason()
    if reason is not None:
        return ProbeResult(
            source=extractor.source,
            status="skip",
            required=extractor.required,
            detail=reason,
        )
    try:
        extractor.probe()
        return ProbeResult(
            source=extractor.source,
            status="ok",
            required=extractor.required,
            detail="",
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            source=extractor.source,
            status="fail",
            required=extractor.required,
            detail=redact(str(exc)),
        )


def run_healthcheck(extractors: Iterable[type[Extractor]]) -> list[ProbeResult]:
    """Run classify_source for each extractor class in order.

    Instantiates each cls(loader=None); classify_source never touches self.loader.
    Preserves the input iteration order.
    """
    results: list[ProbeResult] = []
    for cls in extractors:
        extractor = cls(loader=None)  # type: ignore[arg-type]
        results.append(classify_source(extractor))
    return results


def format_table(results: list[ProbeResult]) -> str:
    """Format results as a fixed-width 3-column table: SOURCE | STATUS | DETAIL.

    Returns the multi-line string (does not print).
    """
    if not results:
        return "No sources registered."

    # Compute column widths
    header_source = "SOURCE"
    header_status = "STATUS"
    header_detail = "DETAIL"

    col_source = max(len(header_source), *(len(r.source) for r in results))
    col_status = max(len(header_status), *(len(r.status) for r in results))

    sep = f"  {'─' * col_source}  {'─' * col_status}  {'─' * max(len(header_detail), 1)}"

    lines: list[str] = []
    lines.append(f"  {header_source:<{col_source}}  {header_status:<{col_status}}  {header_detail}")
    lines.append(sep)
    for r in results:
        # STATUS column is the bare status; the reason lives in DETAIL only (no duplication).
        lines.append(f"  {r.source:<{col_source}}  {r.status:<{col_status}}  {r.detail}")
    return "\n".join(lines)


def exit_code(results: list[ProbeResult]) -> int:
    """Return 1 iff any result has status=="fail" AND required is True; else 0.

    skip never causes a non-zero exit. An optional source failure is also exit 0.
    """
    for r in results:
        if r.status == "fail" and r.required:
            return 1
    return 0
