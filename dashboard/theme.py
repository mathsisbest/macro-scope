"""Code-defined theme: a Plotly layout template + a little Streamlit CSS.

Everything visual is defined here in code (no BI-tool config), so the look is
version-controlled and consistent across every chart.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

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
