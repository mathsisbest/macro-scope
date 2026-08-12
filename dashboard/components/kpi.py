"""KPI tile helpers.

Provides:
- ``format_value``  — reusable value/delta string formatter.
- ``metric_row``    — renders a row of st.metric tiles, guarded against empty /
                      oversized inputs (0, 1, and 4 tiles all render cleanly).

Delta colour follows the theme semantic tokens (SUCCESS / WARN) via inline CSS so
the green/red signal is consistent with PALETTE['up'] / PALETTE['down'] and requires
no inline hex strings.
"""

from __future__ import annotations

import math
from typing import Literal

import streamlit as st
from dashboard.theme import SUCCESS, WARN

# ---------------------------------------------------------------------------
# Public helper — reusable value / delta formatter
# ---------------------------------------------------------------------------

_FormatKind = Literal["price", "percent", "spread", "plain"]

_MAX_TILES = 8  # guard against an absurdly wide layout


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
# Delta-colour CSS injection (theme-token based, no inline hex)
# ---------------------------------------------------------------------------


def _delta_css() -> None:
    """Inject CSS so st.metric delta text uses theme UP/DOWN tokens.

    Streamlit uses ``data-testid="stMetricDelta"`` on the delta span and adds
    a child element with class ``positive`` / ``negative``.  We map those to the
    PALETTE up/down hexes via the semantic SUCCESS / WARN tokens — no inline hex
    literals anywhere else.
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
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# metric_row — renders a guarded row of KPI tiles
# ---------------------------------------------------------------------------


import pandas as pd

def sparkline_metric(label: str, value: str, delta: str, history: pd.Series) -> None:
    # Handle optional/empty values cleanly
    val_str = str(value) if value is not None else "—"
    delta_str = str(delta) if delta is not None else ""
    
    if history is None or history.empty:
        spark_html = ""
    else:
        hist_vals = history.dropna().tolist()
        if not hist_vals:
            spark_html = ""
        else:
            min_val = min(hist_vals)
            max_val = max(hist_vals)
            rng = max_val - min_val if max_val != min_val else 1
            pts = []
            width = 120
            height = 30
            for i, v in enumerate(hist_vals):
                x = i * (width / max(1, len(hist_vals) - 1))
                y = height - ((v - min_val) / rng) * height
                pts.append(f"{x},{y}")
            pts_str = " ".join(pts)
            
            # Default color logic based on delta
            color = SUCCESS if ("+" in delta_str or not delta_str.startswith("-")) else WARN
            
            # Contextual thresholds
            if "VIX" in label.upper():
                try:
                    val_float = float(val_str.strip("$% pp").replace(",", ""))
                    if val_float < 15: color = SUCCESS
                    elif val_float <= 25: color = "#ffb454" # yellow
                    else: color = WARN
                except ValueError:
                    pass
            elif "spread" in label.lower() or "yield curve" in label.lower():
                try:
                    val_float = float(val_str.strip("$% pp").replace(",", "").replace("+", ""))
                    if val_float < 0: color = WARN
                    else: color = SUCCESS
                except ValueError:
                    pass
                
            spark_html = f"""
            <svg width="{width}" height="{height}" style="margin-top:8px; overflow:visible; display:block;">
                <polyline fill="none" stroke="{color}" stroke-width="2" points="{pts_str}" />
            </svg>
            """
    
    delta_color = SUCCESS if ("+" in delta_str or not delta_str.startswith("-")) else WARN
    delta_html = f'<div style="font-size: 14px; margin-top: 4px; color: {delta_color};">{delta_str}</div>' if delta_str else ""
    
    html = f"""
    <div data-testid="stMetric" style="padding: 14px 16px; border-radius: 12px; border: 1px solid #2a2f3a; background: #161a25; min-height: 120px;">
        <div data-testid="stMetricLabel" style="color: #9aa0aa; font-size: 14px;">{label}</div>
        <div data-testid="stMetricValue" style="color: #e6e6e6; font-size: 24px; font-weight: bold; line-height: 1.2;">{val_str}</div>
        {delta_html}
        {spark_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def metric_row(items: list[dict]) -> None:
    """Render a row of KPI tiles with optional sparklines."""
    if not items:
        return

    truncated = False
    if len(items) > _MAX_TILES:
        items = list(items[:_MAX_TILES])
        truncated = True

    _delta_css()

    _ROW_SIZE = 4
    for start in range(0, len(items), _ROW_SIZE):
        chunk = items[start : start + _ROW_SIZE]
        cols = st.columns(len(chunk))
        for col, item in zip(cols, chunk, strict=False):
            with col:
                if "history" in item:
                    sparkline_metric(
                        label=item.get("label", ""),
                        value=item.get("value", "—"),
                        delta=item.get("delta", ""),
                        history=item["history"]
                    )
                else:
                    st.metric(
                        label=item.get("label", ""),
                        value=item.get("value", "—"),
                        delta=item.get("delta"),
                    )

    if truncated:
        st.caption(
            f"Showing first {_MAX_TILES} of the available KPI tiles. "
            "Reduce the number of items passed to `metric_row` to display all."
        )
