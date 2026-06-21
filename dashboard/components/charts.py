"""Plotly chart builders — all styling routed through theme.style_fig."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dashboard.theme import PALETTE, style_fig


def price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=df["date"], y=df["close"], name=symbol, line=dict(color=PALETTE["accent"]))
    if "ma_50" in df:
        fig.add_scatter(
            x=df["date"],
            y=df["ma_50"],
            name="50d MA",
            line=dict(color=PALETTE["muted"], dash="dash"),
        )
    fig.update_layout(title=f"{symbol} — price & 50d moving average")
    return style_fig(fig)


def vol_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["date"],
        y=df["vol_20d"],
        name="20d vol",
        fill="tozeroy",
        line=dict(color=PALETTE["series"][2]),
    )
    fig.update_layout(title=f"{symbol} — rolling 20-day volatility")
    return style_fig(fig, height=260)


def crypto_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["ts"], y=df["price_usd"], name=symbol, line=dict(color=PALETTE["series"][1])
    )
    fig.update_layout(title=f"{symbol.title()} — intraday price (USD)")
    return style_fig(fig)


def macro_chart(df: pd.DataFrame, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=df["date"], y=df["value"], name=label, line=dict(color=PALETTE["accent"]))
    fig.update_layout(title=label)
    return style_fig(fig)


def yield_curve_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["date"],
        y=df["yield_curve_10y_2y"],
        name="10Y-2Y spread",
        line=dict(color=PALETTE["series"][3]),
    )
    fig.add_hline(y=0, line_color=PALETTE["down"], line_dash="dot")
    fig.update_layout(title="Yield-curve spread (10Y − 2Y) — inversion below 0")
    return style_fig(fig)


def forecast_bar(metrics: pd.DataFrame, symbol: str) -> go.Figure:
    m = metrics[metrics["symbol"] == symbol].set_index("metric")["value"]
    fig = go.Figure()
    fig.add_bar(
        x=["Model", "Baseline"],
        y=[m.get("dir_acc", 0), m.get("baseline_dir_acc", 0)],
        marker_color=[PALETTE["up"], PALETTE["muted"]],
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(title=f"{symbol} — directional accuracy vs baseline")
    return style_fig(fig, height=300)


def regime_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    colors = {"Low": PALETTE["up"], "Medium": PALETTE["series"][2], "High": PALETTE["down"]}
    fig = go.Figure()
    for regime, grp in df.groupby("regime"):
        fig.add_scatter(
            x=grp["date"],
            y=grp["vol_20d"],
            mode="markers",
            name=str(regime),
            marker=dict(color=colors.get(str(regime), PALETTE["accent"]), size=5),
        )
    fig.update_layout(title=f"{symbol} — volatility regimes")
    return style_fig(fig, height=300)
