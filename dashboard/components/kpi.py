"""KPI tile helpers.

Provides:
- ``format_value``  — reusable value/delta string formatter.
- ``sparkline_points`` / ``sma`` / ``classify_threshold`` / ``threshold_indicator``
                    — pure helpers for the sparkline + contextual-threshold upgrade.
- ``sparkline_chart`` — tiny Plotly sparkline figure (styled via theme.style_fig).
- ``metric_row``    — renders a row of st.metric tiles, guarded against empty /
                      oversized inputs (0, 1, and 4 tiles all render cleanly).
                      Each tile MAY carry an optional ``"sparkline"`` series and/or a
                      ``"threshold"`` dict — both are additive, so existing callers
                      keep working unchanged.

Delta colour follows the theme semantic tokens (SUCCESS / WARN) via inline CSS so
the green/red signal is consistent with PALETTE['up'] / PALETTE['down'] and requires
no inline hex strings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypedDict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.theme import (
    HEIGHT_SPARKLINE,
    PALETTE,
    PLOTLY_CONFIG,
    SUCCESS,
    WARN,
    style_fig,
)

# ---------------------------------------------------------------------------
# Public helper — reusable value / delta formatter
# ---------------------------------------------------------------------------

_FormatKind = Literal["price", "percent", "spread", "plain"]

#: One KPI tile: pre-formatted ``label`` + ``value`` string (+ optional ``delta``). All keys
#: optional because ``metric_row`` tolerates missing keys (it falls back to its own defaults).
#: Functional form (not class syntax) because mypy runs in python_version=3.10 mode, where
#: ``total=False`` / ``NotRequired`` class syntax is unavailable.
KpiItem = TypedDict(  # noqa: UP013
    "KpiItem",
    {
        "label": str,
        "value": str,
        "delta": str,
    },
    total=False,
)

_MAX_TILES: int = 8  # guard against an absurdly wide layout

#: Default number of most-recent points used for a KPI sparkline.
_SPARKLINE_WINDOW = 90
_ThresholdRelation = Literal["above", "below", "at"]
_GoodWhen = Literal["above", "below"]
_ThresholdModifier = Literal["good", "bad", "neutral"]


def yield_curve_spread_pick(mm: pd.DataFrame) -> tuple[str, str] | None:
    """Choose the recession-risk yield-curve spread column + its honest label.

    Prefers the canonical 10Y−3M spread (NY Fed / Estrella-Mishkin — the inversion investors
    watch for recession risk, and what the recession-risk panel uses); falls back to 10Y−2Y
    when the 3M series is unavailable (e.g. a snapshot taken before the 10Y−3M column
    existed). Returns ``(column, label)`` or ``None`` when neither series has data — the
    label ALWAYS matches the chosen column (a 2Y fallback tile must never claim to be the
    10Y−3M spread). Pure + unit-tested.
    """
    if mm is None or mm.empty:
        return None
    if "yield_curve_10y_3m" in mm.columns and mm["yield_curve_10y_3m"].notna().any():
        return "yield_curve_10y_3m", "10Y−3M spread"
    if mm.get("yield_curve_10y_2y", pd.Series(dtype=float)).notna().any():
        return "yield_curve_10y_2y", "10Y−2Y spread"
    return None


def format_value(
    raw: float | int | str | None,
    kind: _FormatKind = "plain",
    *,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Return a display-ready string for *raw* according to *kind*.

    Parameters
    ----------
    raw:
        The underlying numeric (or already-string) value.
    kind:
        ``"price"``   → ``$1,234.56`` (comma-separated, 2 dp)
        ``"percent"`` → ``+1.23%``    (sign-forced, 2 dp)
        ``"spread"``  → ``+1.23 pp``  (sign-forced, 2 dp, " pp" suffix)
        ``"plain"``   → ``str(raw)``  (pass-through; honours *prefix*/*suffix*)
    prefix:
        Prepended *before* the formatted number (e.g. ``"$"``).  Ignored for
        ``"price"`` (which embeds its own ``$``) and ``"percent"``/``"spread"``.
    suffix:
        Appended *after* the formatted number (ignored for ``"percent"``/
        ``"spread"`` which embed their own units).

    Returns
    -------
    str
        A human-readable string, or ``"—"`` when *raw* is ``None``, ``NaN``, or
        infinite (these are missing/undefined, not real values — rendering them
        as ``"$nan"`` / ``"+inf%"`` would look valid but isn't).
    """
    if raw is None:
        return "—"  # em-dash for missing data

    if isinstance(raw, str):
        # Already formatted by the caller; just honour prefix/suffix.
        return f"{prefix}{raw}{suffix}"

    val = float(raw)

    if math.isnan(val) or math.isinf(val):
        return "—"  # NaN / inf are not real values — match the None convention

    if kind == "price":
        return f"${val:,.2f}"
    if kind == "percent":
        return f"{val:+.2f}%"
    if kind == "spread":
        return f"{val:+.2f} pp"
    # plain
    return f"{prefix}{val}{suffix}"


# ---------------------------------------------------------------------------
# Sparkline + contextual-threshold helpers (pure, unit-tested)
# ---------------------------------------------------------------------------


def sparkline_points(
    series: Sequence[float] | pd.Series | None, window: int = _SPARKLINE_WINDOW
) -> list[float]:
    """Return the ``window`` most-recent finite values of *series* for a sparkline.

    Missing data is dropped (NaN / ±inf / None), so a series with holes renders
    without gaps and the most recent valid observations are kept. Returns ``[]``
    when *series* is ``None``/empty or carries no finite values — callers render
    no sparkline rather than a misleading flat line.
    """
    if series is None:
        return []
    values = series if isinstance(series, pd.Series) else pd.Series(list(series))
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    cleaned = cleaned[cleaned.map(math.isfinite)]
    if cleaned.empty:
        return []
    return [float(v) for v in cleaned.tail(window)]


def sma(series: Sequence[float] | pd.Series | None, window: int = 20) -> float | None:
    """Trailing simple moving average of the ``window`` most-recent finite values.

    ``None`` when the series has no finite values — a missing baseline, so the
    threshold indicator must render nothing (never a fake number). Uses the same
    finite-slicing as :func:`sparkline_points` so the two always agree.
    """
    points = sparkline_points(series, window=window)
    if not points:
        return None
    return sum(points) / len(points)


def classify_threshold(
    current: float | None,
    reference: float | None,
    *,
    tolerance: float = 0.0,
) -> _ThresholdRelation | None:
    """Classify *current* against *reference*: ``"above"`` / ``"below"`` / ``"at"``.

    ``"at"`` means within ±*tolerance* of the reference. Returns ``None`` when
    either side is missing or non-finite — no data, no (misleading) verdict.
    """
    if current is None or reference is None:
        return None
    if not math.isfinite(float(current)) or not math.isfinite(float(reference)):
        return None
    if current > reference + tolerance:
        return "above"
    if current < reference - tolerance:
        return "below"
    return "at"


def threshold_indicator(
    current: float | None,
    reference: float | None,
    *,
    good_when: _GoodWhen = "above",
    tolerance: float = 0.0,
    label: str = "threshold",
) -> dict | None:
    """One display-ready threshold verdict for a KPI tile, or ``None`` when uncomputable.

    Returns a ``dict`` with:
      - ``"relation"`` — ``"above"`` / ``"below"`` / ``"at"`` (see :func:`classify_threshold`)
      - ``"text"``     — e.g. ``"▲ above 20d avg"`` (arrow + relation + *label*)
      - ``"color"``    — theme token: SUCCESS / WARN / muted per *good_when*
      - ``"modifier"`` — ``"good"`` / ``"bad"`` / ``"neutral"`` CSS class suffix

    *good_when* encodes whether being ABOVE the reference is the good direction
    (a price vs its moving average: yes) or the bad one (the yield-curve spread
    vs its zero/inversion threshold: no — below zero is the risk signal).
    """
    relation = classify_threshold(current, reference, tolerance=tolerance)
    if relation is None:
        return None
    arrows: dict[_ThresholdRelation, str] = {"above": "▲", "below": "▼", "at": "≈"}
    text = f"{arrows[relation]} {relation} {label}"
    if relation == "at":
        modifier: _ThresholdModifier = "neutral"
        color: str = PALETTE["muted"]
    elif (relation == "above") == (good_when == "above"):
        modifier = "good"
        color = SUCCESS
    else:
        modifier = "bad"
        color = WARN
    return {"relation": relation, "text": text, "color": color, "modifier": modifier}


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba(...)`` string at *alpha* (sparkline fill)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def sparkline_chart(points: Sequence[float], *, color: str = PALETTE["accent"]) -> go.Figure:
    """Tiny filled line sparkline for a KPI tile, house-styled via ``theme.style_fig``.

    Axes and legend are hidden (the sparkline is a trend hint, not a chart to
    interrogate); the line takes *color* (default accent) so callers can tint it
    from a threshold verdict. Empty *points* still yields a valid, empty figure.
    """
    fig = go.Figure()
    if points:
        fig.add_scatter(
            y=list(points),
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=_rgba(color, 0.15),
            hoverinfo="skip",
        )
    fig.update_layout(
        showlegend=False,
        hovermode=False,
        margin=dict(l=0, r=0, t=4, b=0),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height=HEIGHT_SPARKLINE)


# ---------------------------------------------------------------------------
# KPI-card CSS injection (theme-token based, no inline hex)
# ---------------------------------------------------------------------------


def _kpi_css() -> None:
    """Inject CSS so KPI deltas and threshold captions use theme tokens.

    Streamlit uses ``data-testid="stMetricDelta"`` on the delta span and adds
    a child element with class ``positive`` / ``negative``.  We map those to the
    PALETTE up/down hexes via the semantic SUCCESS / WARN tokens — no inline hex
    literals anywhere else.  Threshold captions render as ``span.kpi-threshold``
    with a ``--good`` / ``--bad`` / ``--neutral`` modifier mapped to the same
    SUCCESS / WARN / muted tokens.
    """
    st.markdown(
        f"""
        <style>
        [data-testid="stMetricDelta"] svg {{ display: none; }}
        [data-testid="stMetricDelta"] > div[class*="positive"] {{
            color: {SUCCESS} !important;
        }}
        [data-testid="stMetricDelta"] > div[class*="negative"] {{
            color: {WARN} !important;
        }}
        .kpi-threshold {{
            display: block;
            font-size: 0.78rem;
            margin-top: 4px;
        }}
        .kpi-threshold--good {{ color: {SUCCESS}; }}
        .kpi-threshold--bad {{ color: {WARN}; }}
        .kpi-threshold--neutral {{ color: {PALETTE["muted"]}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# metric_row — renders a guarded row of KPI tiles
# ---------------------------------------------------------------------------


def metric_row(items: list[KpiItem]) -> None:
    """Render a row of ``st.metric`` tiles.

    Each *item* is a ``dict`` with keys:
      - ``"label"``  (str) — tile label
      - ``"value"``  (str) — formatted value string
      - ``"delta"``  (str, optional) — delta string passed to ``st.metric``
      - ``"sparkline"`` (optional) — ``pd.Series`` / list of floats: recent history
        rendered as a small house-styled sparkline under the value.
      - ``"threshold"`` (optional) — ``dict`` with ``"reference"`` (float),
        ``"good_when"`` (``"above"``|``"below"``, default ``"above"``),
        ``"label"`` (str, default ``"threshold"``), ``"tolerance"`` (float,
        default 0.0) and optional ``"current"`` (float — defaults to the last
        finite sparkline point).  Renders a contextual arrow caption (e.g.
        ``"▲ above 20d avg"``) coloured by the SUCCESS/WARN/muted theme tokens.

    Both ``"sparkline"`` and ``"threshold"`` are **additive** — items without
    them render exactly as before, so existing callers are unaffected.

    Guard rules
    -----------
    - **0 items** → renders nothing (no empty columns).
    - **1–``_MAX_TILES`` items** → renders a single ``st.columns`` row.
    - **> ``_MAX_TILES`` items** → only the first ``_MAX_TILES`` tiles are shown
      and a caption warns that the display was truncated.  This prevents an
      absurdly wide layout on small screens.

    Delta colour
    ------------
    CSS is injected once per call so positive deltas use ``theme.SUCCESS``
    (PALETTE['up'] = #27c08a) and negative deltas use ``theme.WARN``
    (PALETTE['down'] = #ff5d6c).  No inline hex strings are used.
    """
    if not items:
        return

    truncated = False
    if len(items) > _MAX_TILES:
        items = list(items[:_MAX_TILES])
        truncated = True

    _kpi_css()

    # Chunk into rows of at most 4 so the layout stays readable on narrow screens.
    _ROW_SIZE = 4
    for start in range(0, len(items), _ROW_SIZE):
        chunk = items[start : start + _ROW_SIZE]
        cols = st.columns(len(chunk))
        for col, item in zip(cols, chunk, strict=False):
            col.metric(
                label=item.get("label", ""),
                value=item.get("value", "—"),
                delta=item.get("delta"),
            )
            points = (
                sparkline_points(item["sparkline"]) if item.get("sparkline") is not None else []
            )
            threshold_cfg = item.get("threshold")
            indicator = None
            if threshold_cfg:
                current = threshold_cfg.get("current")
                if current is None and points:
                    current = points[-1]
                indicator = threshold_indicator(
                    current,
                    threshold_cfg.get("reference"),
                    good_when=threshold_cfg.get("good_when", "above"),
                    tolerance=threshold_cfg.get("tolerance", 0.0),
                    label=threshold_cfg.get("label", "threshold"),
                )
            if points:
                col.plotly_chart(
                    sparkline_chart(
                        points,
                        color=indicator["color"] if indicator else PALETTE["accent"],
                    ),
                    config=PLOTLY_CONFIG,
                )
            if indicator:
                col.markdown(
                    f'<span class="kpi-threshold kpi-threshold--{indicator["modifier"]}">'
                    f"{indicator['text']}</span>",
                    unsafe_allow_html=True,
                )

    if truncated:
        st.caption(
            f"Showing first {_MAX_TILES} of the available KPI tiles. "
            "Reduce the number of items passed to `metric_row` to display all."
        )
