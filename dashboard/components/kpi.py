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


def metric_row(items: list[dict]) -> None:
    """Render a row of ``st.metric`` tiles.

    Each *item* is a ``dict`` with keys:
      - ``"label"``  (str) — tile label
      - ``"value"``  (str) — formatted value string
      - ``"delta"``  (str, optional) — delta string passed to ``st.metric``

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

    _delta_css()

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

    if truncated:
        st.caption(
            f"Showing first {_MAX_TILES} of the available KPI tiles. "
            "Reduce the number of items passed to `metric_row` to display all."
        )
