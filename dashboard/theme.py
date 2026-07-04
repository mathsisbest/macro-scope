"""Code-defined theme: a Plotly layout template + a little Streamlit CSS.

Everything visual is defined here in code (no BI-tool config), so the look is
version-controlled and consistent across every chart.

WCAG-AA contrast audit (all vs bg #0e1117 / vs panel #161a25):
  text  #e6e6e6 → 15.14:1 / 13.92:1  ✓ AA-normal
  muted #9aa0aa →  7.18:1 /  6.61:1  ✓ AA-normal
  up    #27c08a →  8.09:1 /  7.44:1  ✓ AA-normal
  down  #ff5d6c →  6.33:1 /  5.82:1  ✓ AA-normal
  accent #4f9dff → 6.84:1 /  6.29:1  ✓ AA-normal
All small-text pairs pass WCAG-AA (≥ 4.5:1). No compliant-token additions needed.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Primitive palette — IMMUTABLE (existing keys + hexes must never change).
# charts.py imports PALETTE['accent'], ['series'], ['up'], ['down'], ['muted'].
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#0e1117",
    "panel": "#161a25",
    "grid": "#2a2f3a",
    "text": "#e6e6e6",
    "muted": "#9aa0aa",
    "accent": "#4f9dff",
    "up": "#27c08a",
    "down": "#ff5d6c",
    "series": ["#4f9dff", "#27c08a", "#ffb454", "#c678dd", "#ff5d6c", "#56b6c2"],
}

# ---------------------------------------------------------------------------
# Semantic tokens — additive layer on top of PALETTE.
# Use these in new UI code; never add inline hex strings.
# ---------------------------------------------------------------------------

# Status colours (map intent → primitive)
SUCCESS: str = PALETTE["up"]  # positive / cleared / passing
WARN: str = PALETTE["down"]  # warning / failed / at-risk
INFO: str = PALETTE["accent"]  # informational highlight

# Named series colour tokens — use these instead of PALETTE["series"][<index>] in charts.
# Ordering mirrors PALETTE["series"] so the rendered colour is stable.
SERIES_PRICE: str = PALETTE["series"][0]  # #4f9dff — price / accent line
SERIES_RETURN: str = PALETTE["series"][1]  # #27c08a — return / crypto / positive series
SERIES_VOL: str = PALETTE["series"][2]  # #ffb454 — volatility / amber
SERIES_YIELD: str = PALETTE["series"][3]  # #c678dd — yield-curve / macro / purple
SERIES_RISK: str = PALETTE["series"][4]  # #ff5d6c — risk / drawdown / danger
SERIES_ALT: str = PALETTE["series"][5]  # #56b6c2 — alternative / teal

# Asset-class colour map — one stable colour per asset class, used for the cross-asset Markets
# view (leaderboard dots + rebased-performance lines). Keyed by the `asset_class` value stored in
# fct_asset_daily ("equities"/"bonds"/"commodities"/"crypto"/"fx"). Distinct from PALETTE["series"]
# (which is the categorical line cycle) so a class reads the same colour everywhere it appears.
# These are decorative dot/line tokens (not body text), so they are exempt from the WCAG-AA
# text-contrast minimums the PALETTE text tokens carry.
ASSET_CLASS_COLORS: dict[str, str] = {
    "equities": "#378ADD",
    "bonds": "#1D9E75",
    "commodities": "#BA7517",
    "crypto": "#7F77DD",
    "fx": "#888780",
}
#: Fallback colour for an asset class not in the map (keeps a new class from rendering colourless).
ASSET_CLASS_FALLBACK: str = PALETTE["muted"]


def asset_class_color(asset_class: str | None) -> str:
    """Stable colour for an ``asset_class`` value, with a muted fallback for unknown classes."""
    return ASSET_CLASS_COLORS.get(str(asset_class or ""), ASSET_CLASS_FALLBACK)


# Chart height scale — consistent height buckets for all figures.
HEIGHT_TALL: int = 400  # full-width hero charts
HEIGHT_DEFAULT: int = 360  # standard figure (matches style_fig default)
HEIGHT_MEDIUM: int = 320  # secondary / paired charts
HEIGHT_SHORT: int = 260  # sparkline / compact supplementary

# Typography
TITLE_SIZE: int = 18  # px — section / card titles
BODY_SIZE: int = 13  # px — body copy (matches style_fig font size)
CAPTION_SIZE: int = 11  # px — captions, badges, footnotes

# Layout / shape
CARD_RADIUS: str = "12px"  # border-radius for metric cards / panels
PANEL_BORDER: str = f"1px solid {PALETTE['grid']}"  # standard card border
CARD_PADDING: str = "14px 16px"  # inner padding for metric cards


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply the house style to any Plotly figure."""
    # A figure title and the horizontal top legend both sit above the plot area; with a tight top
    # margin they collide (the SPY "price & 50d moving average" title overlapped its legend). Give
    # titled figures extra top headroom and pin the title to the very top so the legend sits below
    # it, not on top of it. Title-less figures keep the compact margin.
    has_title = bool(fig.layout.title.text)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=72 if has_title else 40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], size=13),
        colorway=PALETTE["series"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    if has_title:
        fig.update_layout(title=dict(y=1.0, yanchor="top", x=0, xanchor="left", pad=dict(b=6)))
    fig.update_xaxes(gridcolor=PALETTE["grid"], zeroline=False, fixedrange=True)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False, fixedrange=True)
    return fig


# Mobile-safe Plotly config — disables scroll-zoom and hides the floating modebar so
# touch gestures scroll the page instead of manipulating the chart.
PLOTLY_CONFIG: dict = {"scrollZoom": False, "displayModeBar": False}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {PALETTE["bg"]}; }}
        /* Trim Streamlit's tall default top padding (~6rem) so the KPIs and first chart sit
           higher / closer to above-the-fold without crowding the top toolbar. */
        .block-container {{ padding-top: 3rem; }}
        [data-testid="stMetric"] {{
            background: {PALETTE["panel"]}; border: 1px solid {PALETTE["grid"]};
            padding: 14px 16px; border-radius: 12px;
        }}
        [data-testid="stMetricLabel"] {{ color: {PALETTE["muted"]}; }}

        /* ---- Mobile responsive ---- */
        @media (max-width: 768px) {{
            .block-container {{ padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }}
            [data-testid="stMetric"] {{ padding: 10px 12px; }}
            /* Tab bar: scroll horizontally with momentum on iOS */
            [data-testid="stHorizontalBlock"] {{
                overflow-x: auto; -webkit-overflow-scrolling: touch;
            }}
            /* Segmented controls: smaller font + scroll so options don't clip */
            [data-testid="stSegmentedControl"] {{
                overflow-x: auto; -webkit-overflow-scrolling: touch;
            }}
            [data-testid="stSegmentedControl"] label {{ font-size: 0.8rem; white-space: nowrap; }}
            /* Radio buttons: stack vertically when labels are long (portfolio window selector) */
            [data-testid="stRadio"] {{ flex-direction: column !important; }}
            /* Chart pairs: stack vertically so each chart gets full width on phones */
            .chart-pair [data-testid="stHorizontalBlock"] {{
                flex-direction: column !important;
            }}
            .chart-pair [data-testid="stHorizontalBlock"] > div {{
                flex: 0 0 100% !important; max-width: 100% !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
