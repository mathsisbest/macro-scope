"""Regression tests for dashboard/components/kpi.py.

``format_value`` (the original formatter): a non-finite float (NaN / +inf / -inf)
must never render as a real-looking value (``"$nan"``, ``"+inf%"``, ``"-inf pp"`` …).
Such values are missing/undefined, so they must collapse to the same ``"—"`` em-dash
the formatter already uses for ``None`` — the project's "looks valid but isn't"
honesty rule.  A handful of happy-path assertions also lock the formatter contract.

Task 2.8 helpers (sparkline slicing + threshold classification): the pure helpers
behind the KPI sparkline / contextual-threshold upgrade — ``sparkline_points``,
``sma``, ``classify_threshold``, ``threshold_indicator`` and ``sparkline_chart``.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import pytest
from dashboard.components.kpi import (
    _rgba,
    classify_threshold,
    format_value,
    sma,
    sparkline_chart,
    sparkline_points,
    threshold_indicator,
)
from dashboard.theme import HEIGHT_SPARKLINE, PALETTE, SUCCESS, WARN

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


# ---------------------------------------------------------------------------
# Task 2.8 — sparkline_points: slicing of recent history for a sparkline
# ---------------------------------------------------------------------------


def test_sparkline_points_none_or_empty() -> None:
    assert sparkline_points(None) == []
    assert sparkline_points(pd.Series(dtype=float)) == []
    assert sparkline_points([]) == []
    assert sparkline_points([None, float("nan")]) == []


def test_sparkline_points_drops_non_finite_keeps_order() -> None:
    series = pd.Series([1.0, float("nan"), 2.0, float("inf"), 3.0, None, 4.0])
    assert sparkline_points(series) == [1.0, 2.0, 3.0, 4.0]


def test_sparkline_points_takes_last_window_points() -> None:
    series = list(range(100))
    assert sparkline_points(series, window=10) == [
        90.0,
        91.0,
        92.0,
        93.0,
        94.0,
        95.0,
        96.0,
        97.0,
        98.0,
        99.0,
    ]
    assert len(sparkline_points(series, window=90)) == 90
    # Window larger than the series → everything, in order.
    assert sparkline_points(series, window=500) == [float(v) for v in series]


def test_sparkline_points_accepts_plain_list() -> None:
    assert sparkline_points([1, 2, 3], window=2) == [2.0, 3.0]


def test_sparkline_points_gracefully_coerces_errors() -> None:
    """A non-numeric hole must be dropped like any other missing value, not crash."""
    assert sparkline_points(pd.Series(["x", 5.0, 6.0]), window=2) == [5.0, 6.0]


# ---------------------------------------------------------------------------
# Task 2.8 — sma: trailing average of the most-recent finite values
# ---------------------------------------------------------------------------


def test_sma_mean_of_last_window() -> None:
    assert sma([1.0, 2.0, 3.0, 4.0], window=2) == 3.5
    assert sma([1.0, 2.0, 3.0], window=10) == 2.0  # window > series → all points


def test_sma_ignores_missing_values() -> None:
    assert sma(pd.Series([float("nan"), 2.0, 4.0, float("inf")]), window=2) == 3.0


def test_sma_none_when_no_finite_data() -> None:
    assert sma(None) is None
    assert sma([]) is None
    assert sma([float("nan"), float("inf")]) is None


# ---------------------------------------------------------------------------
# Task 2.8 — classify_threshold: above / below / at
# ---------------------------------------------------------------------------


def test_classify_threshold_above_below_at() -> None:
    assert classify_threshold(1.1, 1.0) == "above"
    assert classify_threshold(0.9, 1.0) == "below"
    assert classify_threshold(1.0, 1.0) == "at"


def test_classify_threshold_tolerance_band() -> None:
    assert classify_threshold(1.04, 1.0, tolerance=0.05) == "at"
    assert classify_threshold(1.06, 1.0, tolerance=0.05) == "above"
    assert classify_threshold(0.94, 1.0, tolerance=0.05) == "below"
    # Tolerance band boundary is inclusive of the limit.
    assert classify_threshold(1.05, 1.0, tolerance=0.05) == "at"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_classify_threshold_missing_or_non_finite_is_none(bad: float | None) -> None:
    assert classify_threshold(bad, 1.0) is None
    assert classify_threshold(1.0, bad) is None
    assert classify_threshold(bad, bad) is None


# ---------------------------------------------------------------------------
# Task 2.8 — threshold_indicator: display verdict + theme-token colouring
# ---------------------------------------------------------------------------


def test_threshold_indicator_good_when_above() -> None:
    above = threshold_indicator(1.1, 1.0, good_when="above", label="20d avg")
    assert above is not None
    assert above["relation"] == "above"
    assert above["text"] == "▲ above 20d avg"
    assert above["color"] == SUCCESS
    assert above["modifier"] == "good"

    below = threshold_indicator(0.9, 1.0, good_when="above")
    assert below is not None
    assert below["relation"] == "below"
    assert below["color"] == WARN
    assert below["modifier"] == "bad"


def test_threshold_indicator_good_when_below_inverts_colours() -> None:
    """Below the reference is the GOOD direction (e.g. high vol / high risk)."""
    good = threshold_indicator(0.5, 1.0, good_when="below", label="risk limit")
    assert good is not None
    assert good["text"] == "▼ below risk limit"
    assert good["color"] == SUCCESS
    assert good["modifier"] == "good"

    bad = threshold_indicator(1.5, 1.0, good_when="below")
    assert bad is not None
    assert bad["color"] == WARN
    assert bad["modifier"] == "bad"


def test_threshold_indicator_at_is_neutral() -> None:
    neutral = threshold_indicator(1.0, 1.0, good_when="above", tolerance=0.01)
    assert neutral is not None
    assert neutral["relation"] == "at"
    assert neutral["color"] == PALETTE["muted"]
    assert neutral["modifier"] == "neutral"
    assert neutral["text"] == "≈ at threshold"


def test_threshold_indicator_missing_data_is_none() -> None:
    assert threshold_indicator(None, 1.0) is None
    assert threshold_indicator(1.0, None) is None
    assert threshold_indicator(1.0, float("nan")) is None


# ---------------------------------------------------------------------------
# Task 2.8 — sparkline_chart: house-styled mini figure
# ---------------------------------------------------------------------------


def test_sparkline_chart_plots_points_with_default_accent() -> None:
    fig = sparkline_chart([1.0, 2.0, 3.0])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [1.0, 2.0, 3.0]
    assert fig.layout.height == HEIGHT_SPARKLINE
    assert fig.data[0].line.color == PALETTE["accent"]
    assert not fig.layout.showlegend  # sparklines never carry a legend


def test_sparkline_chart_applies_verdict_colour() -> None:
    fig = sparkline_chart([1.0, 2.0], color=SUCCESS)
    assert fig.data[0].line.color == SUCCESS
    assert fig.data[0].fillcolor == _rgba(SUCCESS, 0.15)


def test_sparkline_chart_empty_points_yields_valid_figure() -> None:
    fig = sparkline_chart([])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


# ---------------------------------------------------------------------------
# _rgba — hex → rgba conversion used by the sparkline fill
# ---------------------------------------------------------------------------


def test_rgba_converts_hex() -> None:
    assert _rgba("#27c08a", 0.15) == "rgba(39, 192, 138, 0.15)"
    assert _rgba("4f9dff", 0.5) == "rgba(79, 157, 255, 0.5)"
