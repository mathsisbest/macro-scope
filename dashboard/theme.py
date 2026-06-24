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
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], size=13),
        colorway=PALETTE["series"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False)
    return fig


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {PALETTE["bg"]}; }}
        [data-testid="stMetric"] {{
            background: {PALETTE["panel"]}; border: 1px solid {PALETTE["grid"]};
            padding: 14px 16px; border-radius: 12px;
        }}
        [data-testid="stMetricLabel"] {{ color: {PALETTE["muted"]}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
