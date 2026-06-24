"""Plotly chart builders — all styling routed through theme.style_fig."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dashboard.theme import (
    HEIGHT_DEFAULT,
    HEIGHT_MEDIUM,
    HEIGHT_SHORT,
    HEIGHT_TALL,
    PALETTE,
    SERIES_ALT,
    SERIES_RETURN,
    SERIES_VOL,
    SERIES_YIELD,
    style_fig,
)

# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------

_TITLE_FONT = dict(size=15, color=PALETTE["text"])
_AXIS_FONT = dict(size=12, color=PALETTE["muted"])
_LEGEND_MAX_ENTRIES = 8  # beyond this, legend moves inside to prevent overflow


def _apply_axis_fonts(fig: go.Figure) -> None:
    """Consistent axis tick + title fonts on every figure."""
    fig.update_xaxes(tickfont=_AXIS_FONT, title_font=_AXIS_FONT)
    fig.update_yaxes(tickfont=_AXIS_FONT, title_font=_AXIS_FONT)


def _overflow_legend(fig: go.Figure, n_traces: int) -> None:
    """Push legend inside the plot area when there are many traces to avoid horizontal overflow."""
    if n_traces > _LEGEND_MAX_ENTRIES:
        fig.update_layout(
            legend=dict(
                orientation="v",
                x=1.01,
                y=1,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
            )
        )


def _guard_yrange(fig: go.Figure, series: pd.Series, pad: float = 0.05) -> None:
    """Widen the y-axis range by `pad` fraction when the data is purely non-negative
    (avoids the chart clipping zero line) or purely non-positive (avoids clipping at zero)."""
    if series.empty:
        return
    lo, hi = float(series.min()), float(series.max())
    span = hi - lo or 1.0
    if lo >= 0:
        fig.update_yaxes(range=[max(0.0, lo - span * pad), hi + span * pad])
    elif hi <= 0:
        fig.update_yaxes(range=[lo - span * pad, min(0.0, hi + span * pad)])


# ---------------------------------------------------------------------------
# Markets tab
# ---------------------------------------------------------------------------


def price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["date"],
        y=df["close"],
        name=symbol,
        line=dict(color=PALETTE["accent"]),
    )
    if "ma_50" in df.columns:
        fig.add_scatter(
            x=df["date"],
            y=df["ma_50"],
            name="50d MA",
            line=dict(color=PALETTE["muted"], dash="dash"),
        )
    fig.update_layout(
        title=dict(text=f"{symbol} — price & 50d moving average", font=_TITLE_FONT),
    )
    _apply_axis_fonts(fig)
    if not df.empty and "close" in df.columns:
        _guard_yrange(fig, df["close"])
    return style_fig(fig, height=HEIGHT_DEFAULT)


def vol_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["date"],
        y=df["vol_20d"],
        name="20d vol",
        fill="tozeroy",
        line=dict(color=SERIES_VOL),
    )
    fig.update_layout(
        title=dict(text=f"{symbol} — rolling 20-day volatility", font=_TITLE_FONT),
    )
    _apply_axis_fonts(fig)
    if not df.empty and "vol_20d" in df.columns:
        _guard_yrange(fig, df["vol_20d"])
    return style_fig(fig, height=HEIGHT_SHORT)


def crypto_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["ts"],
        y=df["price_usd"],
        name=symbol,
        line=dict(color=SERIES_RETURN),
    )
    fig.update_layout(
        title=dict(text=f"{symbol.title()} — intraday price (USD)", font=_TITLE_FONT),
    )
    _apply_axis_fonts(fig)
    if not df.empty and "price_usd" in df.columns:
        _guard_yrange(fig, df["price_usd"])
    return style_fig(fig, height=HEIGHT_DEFAULT)


# ---------------------------------------------------------------------------
# Macro tab
# ---------------------------------------------------------------------------


def macro_chart(df: pd.DataFrame, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["date"],
        y=df["value"],
        name=label,
        line=dict(color=PALETTE["accent"]),
    )
    fig.update_layout(title=dict(text=label, font=_TITLE_FONT))
    _apply_axis_fonts(fig)
    return style_fig(fig, height=HEIGHT_DEFAULT)


def yield_curve_chart(df: pd.DataFrame) -> go.Figure:
    """Yield-curve spread (10Y − 2Y).  Inversion belt shaded red below zero."""
    fig = go.Figure()
    fig.add_scatter(
        x=df["date"],
        y=df["yield_curve_10y_2y"],
        name="10Y-2Y spread",
        line=dict(color=SERIES_YIELD),
    )
    fig.add_hline(y=0, line_color=PALETTE["down"], line_dash="dot")
    fig.update_layout(
        title=dict(
            text="Yield-curve spread (10Y − 2Y) — inversion below 0",
            font=_TITLE_FONT,
        ),
    )
    _apply_axis_fonts(fig)
    # Y-range guard: keep the zero-line visible with symmetric padding
    if not df.empty and "yield_curve_10y_2y" in df.columns:
        series = df["yield_curve_10y_2y"].dropna()
        if not series.empty:
            lo, hi = float(series.min()), float(series.max())
            span = max(abs(lo), abs(hi), 0.5)
            fig.update_yaxes(range=[-span * 1.15, span * 1.15])
    return style_fig(fig, height=HEIGHT_DEFAULT)


# ---------------------------------------------------------------------------
# ML tab
# ---------------------------------------------------------------------------


def forecast_bar(metrics: pd.DataFrame, symbol: str) -> go.Figure:
    m = metrics[metrics["symbol"] == symbol].set_index("metric")["value"]
    fig = go.Figure()
    fig.add_bar(
        x=["Model", "Baseline"],
        y=[m.get("dir_acc", 0), m.get("baseline_dir_acc", 0)],
        marker_color=[PALETTE["up"], PALETTE["muted"]],
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(
        title=dict(
            text=f"{symbol} — directional accuracy vs baseline",
            font=_TITLE_FONT,
        ),
    )
    _apply_axis_fonts(fig)
    return style_fig(fig, height=HEIGHT_MEDIUM)


def regime_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    colors = {
        "Low": PALETTE["up"],
        "Medium": SERIES_VOL,
        "High": PALETTE["down"],
    }
    fig = go.Figure()
    for regime, grp in df.groupby("regime"):
        fig.add_scatter(
            x=grp["date"],
            y=grp["vol_20d"],
            mode="markers",
            name=str(regime),
            marker=dict(color=colors.get(str(regime), PALETTE["accent"]), size=5),
        )
    fig.update_layout(
        title=dict(text=f"{symbol} — volatility regimes", font=_TITLE_FONT),
    )
    _apply_axis_fonts(fig)
    n_traces = df["regime"].nunique() if not df.empty else 0
    _overflow_legend(fig, n_traces)
    return style_fig(fig, height=HEIGHT_MEDIUM)


def ml_gate_chart(gate: pd.DataFrame) -> go.Figure:
    """ML gate over time: weight the forecast earns in mvo_ml (0 = no out-of-sample edge)."""
    fig = go.Figure()
    fig.add_scatter(
        x=gate["date"],
        y=gate["forecast_weight"],
        name="forecast weight (λ)",
        line=dict(color=SERIES_YIELD),
    )
    fig.add_scatter(
        x=gate["date"],
        y=gate["forecast_skill"],
        name="forecast skill",
        line=dict(color=PALETTE["muted"], dash="dash"),
    )
    fig.update_layout(
        title=dict(text="ML gate — forecast weight & skill over time", font=_TITLE_FONT),
    )
    _apply_axis_fonts(fig)
    fig.update_yaxes(rangemode="tozero")
    return style_fig(fig, height=HEIGHT_MEDIUM)


# ---------------------------------------------------------------------------
# Portfolio tab
# ---------------------------------------------------------------------------
_STRATEGY_LABELS = {
    "equal_weight": "Equal weight",
    "inverse_vol": "Inverse vol",
    "risk_parity": "Risk parity",
    "sixty_forty": "60/40 benchmark",
}

# Stable per-strategy named colour tokens (no bare index literals).
_STRATEGY_COLORS: dict[str, str] = {
    "equal_weight": PALETTE["accent"],
    "inverse_vol": SERIES_RETURN,
    "risk_parity": SERIES_VOL,
    "sixty_forty": PALETTE["muted"],  # benchmark is always the muted reference line
    # fallback cycle for any extra strategy keys (additive, uses ALT then YIELD)
    "_fallback": [SERIES_ALT, SERIES_YIELD],
}


def _strategy_line(strategy: str, idx: int) -> dict:
    """Stable per-strategy style; the 60/40 benchmark is a dashed muted reference line."""
    if strategy == "sixty_forty":
        return dict(color=PALETTE["muted"], dash="dash")
    color = _STRATEGY_COLORS.get(strategy)
    if color is None:
        fallback = _STRATEGY_COLORS["_fallback"]
        color = fallback[idx % len(fallback)]
    return dict(color=color)


def _by_strategy(df: pd.DataFrame, column: str) -> go.Figure:
    fig = go.Figure()
    strategies = sorted(df["strategy"].unique())
    for idx, strategy in enumerate(strategies):
        grp = df[df["strategy"] == strategy]
        fig.add_scatter(
            x=grp["date"],
            y=grp[column],
            name=_STRATEGY_LABELS.get(strategy, strategy),
            line=_strategy_line(strategy, idx),
        )
    return fig


def portfolio_cumulative_chart(df: pd.DataFrame) -> go.Figure:
    fig = _by_strategy(df, "cumulative_return")
    fig.update_layout(
        title=dict(
            text="Cumulative return by strategy (vs 60/40 benchmark)",
            font=_TITLE_FONT,
        ),
    )
    fig.update_yaxes(tickformat=".0%")
    _apply_axis_fonts(fig)
    n = df["strategy"].nunique() if not df.empty else 0
    _overflow_legend(fig, n)
    return style_fig(fig, height=HEIGHT_TALL)


def portfolio_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    fig = _by_strategy(df, "drawdown")
    fig.update_layout(
        title=dict(text="Drawdown from running peak", font=_TITLE_FONT),
    )
    fig.update_yaxes(tickformat=".0%")
    # Drawdown is always ≤ 0; keep zero at top, guard the bottom with padding
    if not df.empty and "drawdown" in df.columns:
        lo = float(df["drawdown"].min())
        span = abs(lo) or 0.1
        fig.update_yaxes(range=[lo - span * 0.05, 0])
    _apply_axis_fonts(fig)
    n = df["strategy"].nunique() if not df.empty else 0
    _overflow_legend(fig, n)
    return style_fig(fig, height=HEIGHT_MEDIUM)


def portfolio_sharpe_chart(df: pd.DataFrame) -> go.Figure:
    fig = _by_strategy(df, "rolling_sharpe_252")
    fig.update_layout(
        title=dict(text="Rolling 252-day Sharpe (annualised)", font=_TITLE_FONT),
    )
    _apply_axis_fonts(fig)
    n = df["strategy"].nunique() if not df.empty else 0
    _overflow_legend(fig, n)
    return style_fig(fig, height=HEIGHT_MEDIUM)


def portfolio_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy headline stats for the comparison table (latest values + worst drawdown)."""
    summary = df.groupby("strategy").agg(
        total_return=("cumulative_return", "last"),
        max_drawdown=("drawdown", "min"),
        ann_vol=("daily_return", lambda s: float(s.std() * (252**0.5))),
        sharpe_252=("rolling_sharpe_252", "last"),
    )
    summary.index = [_STRATEGY_LABELS.get(s, s) for s in summary.index]
    summary.index.name = "Strategy"
    return summary.rename(
        columns={
            "total_return": "Total return",
            "max_drawdown": "Max drawdown",
            "ann_vol": "Ann. vol",
            "sharpe_252": "Sharpe (252d)",
        }
    )


def portfolio_scorecard(stats: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy full-sample Sharpe with its bootstrap CI — the risk-adjusted scorecard."""
    out = stats.assign(Strategy=stats["strategy"].map(lambda s: _STRATEGY_LABELS.get(s, s)))
    out = out.set_index("Strategy")[["sharpe", "sharpe_lo", "sharpe_hi"]]
    return out.rename(columns={"sharpe": "Sharpe", "sharpe_lo": "CI low", "sharpe_hi": "CI high"})


def portfolio_pairs_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Sharpe-difference + CI + distinguishability, labelled for display."""

    def lab(strategy: str) -> str:
        return _STRATEGY_LABELS.get(strategy, strategy)

    rows = {
        "Pair": [
            f"{lab(a)} − {lab(b)}"
            for a, b in zip(pairs["strategy_a"], pairs["strategy_b"], strict=True)
        ],
        "Δ Sharpe": pairs["sharpe_diff"].to_numpy(),
        "CI low": pairs["diff_lo"].to_numpy(),
        "CI high": pairs["diff_hi"].to_numpy(),
        "Distinguishable": pairs["distinguishable"].to_numpy(),
    }
    return pd.DataFrame(rows).set_index("Pair")


def distinguishability_verdict(pairs: pd.DataFrame) -> str:
    """One honest line: are any strategy Sharpe differences statistically distinguishable?"""
    if pairs.empty:
        return "Not enough strategies to compare."
    distinct = pairs[pairs["distinguishable"]]
    n = len(pairs)
    if distinct.empty:
        return (
            f"None of the {n} strategy comparisons is statistically distinguishable by Sharpe — "
            "every difference CI includes zero, i.e. the gaps are within noise at this sample size."
        )

    def lab(strategy: str) -> str:
        return _STRATEGY_LABELS.get(strategy, strategy)

    named = ", ".join(f"{lab(r.strategy_a)} vs {lab(r.strategy_b)}" for r in distinct.itertuples())
    return f"{len(distinct)} of {n} comparisons are statistically distinguishable: {named}."


def attribution_chart(attr: pd.DataFrame, strategy: str) -> go.Figure:
    """Horizontal bar of each asset's contribution to a strategy's return (greens up, reds down)."""
    df = attr[attr["strategy"] == strategy].sort_values("contribution_to_return")
    colors = [PALETTE["up"] if v >= 0 else PALETTE["down"] for v in df["contribution_to_return"]]
    fig = go.Figure()
    fig.add_bar(
        x=df["contribution_to_return"],
        y=df["symbol"],
        orientation="h",
        marker_color=colors,
    )
    fig.update_xaxes(tickformat=".1%")
    label = _STRATEGY_LABELS.get(strategy, strategy)
    fig.update_layout(
        title=dict(text=f"{label} — return contribution by asset", font=_TITLE_FONT),
    )
    _apply_axis_fonts(fig)
    return style_fig(fig, height=HEIGHT_MEDIUM + 20)


def regime_sharpe_chart(regime: pd.DataFrame) -> go.Figure:
    """Grouped bars: annualised Sharpe by market volatility regime, one bar per strategy."""
    order = ["Low", "Medium", "High"]
    fig = go.Figure()
    strategies = sorted(regime["strategy"].unique())
    for idx, strategy in enumerate(strategies):
        grp = regime[regime["strategy"] == strategy].set_index("regime").reindex(order)
        fig.add_bar(
            x=order,
            y=grp["ann_sharpe"],
            name=_STRATEGY_LABELS.get(strategy, strategy),
            marker_color=_strategy_line(strategy, idx)["color"],
        )
    fig.update_layout(
        title=dict(text="Annualised Sharpe by market volatility regime", font=_TITLE_FONT),
        barmode="group",
    )
    _apply_axis_fonts(fig)
    n = len(strategies)
    _overflow_legend(fig, n)
    return style_fig(fig, height=HEIGHT_MEDIUM + 20)


def ml_verdict(gate: pd.DataFrame, pairs: pd.DataFrame) -> str:
    """One honest line: did the ML tilt (mvo_ml) beat the historical-mean baseline?"""
    mean_w = float(gate["forecast_weight"].mean()) if not gate.empty else 0.0
    distinguishable = False
    if not pairs.empty:
        pair = {"mvo_histmean", "mvo_ml"}
        match = pairs[pairs.apply(lambda r: {r["strategy_a"], r["strategy_b"]} == pair, axis=1)]
        if not match.empty:
            distinguishable = bool(match["distinguishable"].iloc[0])
    if distinguishable:
        return (
            f"The forecast earned a mean weight of {mean_w:.0%} in the blend, and mvo_ml's Sharpe "
            "is statistically distinguishable from the historical-mean baseline."
        )
    return (
        f"The forecast earned a mean weight of {mean_w:.0%} over the prior — no reliable "
        "out-of-sample edge — so mvo_ml is not statistically distinguishable from the "
        "historical-mean baseline. The ML did not beat the simpler approach."
    )


def btc_effect_chart(effect: pd.DataFrame) -> go.Figure:
    """Per-strategy BTC effect: Sharpe(inc) − Sharpe(ex), 2015 window, with its paired CI."""
    df = effect.sort_values("sharpe_diff")
    colors = [PALETTE["up"] if v >= 0 else PALETTE["down"] for v in df["sharpe_diff"]]
    fig = go.Figure()
    fig.add_bar(
        x=df["sharpe_diff"],
        y=[_STRATEGY_LABELS.get(s, s) for s in df["strategy"]],
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            symmetric=False,
            array=(df["diff_hi"] - df["sharpe_diff"]).to_numpy(),
            arrayminus=(df["sharpe_diff"] - df["diff_lo"]).to_numpy(),
        ),
    )
    fig.add_vline(x=0, line_color=PALETTE["muted"], line_dash="dot")
    fig.update_layout(
        title=dict(
            text="BTC effect on Sharpe (inc − ex, 2015 window) — 90% paired CI",
            font=_TITLE_FONT,
        ),
    )
    _apply_axis_fonts(fig)
    return style_fig(fig, height=HEIGHT_MEDIUM)


def btc_effect_verdict(effect: pd.DataFrame) -> str:
    """One honest line: for how many strategies did adding BTC make a distinguishable difference?"""
    if effect.empty:
        return "BTC effect not computed yet (it needs the 2015 windows)."

    def lab(strategy: str) -> str:
        return _STRATEGY_LABELS.get(strategy, strategy)

    distinct = effect[effect["distinguishable"]]
    n = len(effect)
    if distinct.empty:
        return (
            f"Adding BTC made no statistically distinguishable difference to any of the {n} "
            "strategies' Sharpe over the 2015 window — every paired-difference CI includes zero."
        )
    hurt = distinct[distinct["sharpe_diff"] < 0]
    helped = distinct[distinct["sharpe_diff"] > 0]
    parts = []
    if not hurt.empty:
        parts.append(
            "hurt "
            + ", ".join(f"{lab(r.strategy)} (Δ{r.sharpe_diff:+.2f})" for r in hurt.itertuples())
        )
    if not helped.empty:
        parts.append(
            "helped "
            + ", ".join(f"{lab(r.strategy)} (Δ{r.sharpe_diff:+.2f})" for r in helped.itertuples())
        )
    return (
        f"Adding BTC made a statistically distinguishable difference for {len(distinct)} of {n} "
        f"strategies (same-period paired comparison): it {'; '.join(parts)}."
    )
