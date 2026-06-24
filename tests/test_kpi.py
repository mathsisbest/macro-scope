"""Regression tests for dashboard/components/kpi.py :: format_value.

Primary concern: a non-finite float (NaN / +inf / -inf) must never render as a
real-looking value (``"$nan"``, ``"+inf%"``, ``"-inf pp"`` …).  Such values are
missing/undefined, so they must collapse to the same ``"—"`` em-dash the
formatter already uses for ``None`` — the project's "looks valid but isn't"
honesty rule.  A handful of happy-path assertions also lock the formatter
contract, which previously had no test coverage.
"""

from __future__ import annotations

import math

import pytest
from dashboard.components.kpi import format_value

_KINDS = ["price", "percent", "spread", "plain"]
_NON_FINITE = [float("nan"), float("inf"), float("-inf"), math.nan, math.inf, -math.inf]


# ---------------------------------------------------------------------------
# Core: NaN / inf collapse to the em-dash for every kind.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("bad", _NON_FINITE)
def test_non_finite_renders_em_dash(kind: str, bad: float) -> None:
    """NaN / ±inf must render as "—", never "$nan" / "+inf%" / "-inf pp"."""
    assert format_value(bad, kind) == "—"


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_non_finite_ignores_prefix_suffix(bad: float) -> None:
    """Affixes must not leak a non-finite value back into the output (no "$—%")."""
    assert format_value(bad, "plain", prefix="$", suffix="%") == "—"


# ---------------------------------------------------------------------------
# Contract lock: None and the normal formatting paths still behave.
# ---------------------------------------------------------------------------


def test_none_renders_em_dash() -> None:
    assert format_value(None, "price") == "—"


def test_string_passes_through_with_affixes() -> None:
    assert format_value("n/a", "plain", prefix="[", suffix="]") == "[n/a]"


@pytest.mark.parametrize(
    ("raw", "kind", "expected"),
    [
        (1234.5, "price", "$1,234.50"),
        (1.23, "percent", "+1.23%"),
        (-1.23, "percent", "-1.23%"),
        (1.23, "spread", "+1.23 pp"),
        (-0.5, "spread", "-0.50 pp"),
        (0, "price", "$0.00"),
    ],
)
def test_finite_values_format_normally(raw: float, kind: str, expected: str) -> None:
    assert format_value(raw, kind) == expected
