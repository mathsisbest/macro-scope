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


# --------------------------------------------------------------------------- portfolio
_STRATEGY_LABELS = {
    "equal_weight": "Equal weight",
    "inverse_vol": "Inverse vol",
    "risk_parity": "Risk parity",
    "sixty_forty": "60/40 benchmark",
}


def _strategy_line(strategy: str, idx: int) -> dict:
    """Stable per-strategy style; the 60/40 benchmark is a dashed muted reference line."""
    if strategy == "sixty_forty":
        return dict(color=PALETTE["muted"], dash="dash")
    return dict(color=PALETTE["series"][idx % len(PALETTE["series"])])


def _by_strategy(df: pd.DataFrame, column: str) -> go.Figure:
    fig = go.Figure()
    for idx, strategy in enumerate(sorted(df["strategy"].unique())):
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
    fig.update_layout(title="Cumulative return by strategy (vs 60/40 benchmark)")
    fig.update_yaxes(tickformat=".0%")
    return style_fig(fig)


def portfolio_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    fig = _by_strategy(df, "drawdown")
    fig.update_layout(title="Drawdown from running peak")
    fig.update_yaxes(tickformat=".0%")
    return style_fig(fig, height=300)


def portfolio_sharpe_chart(df: pd.DataFrame) -> go.Figure:
    fig = _by_strategy(df, "rolling_sharpe_252")
    fig.update_layout(title="Rolling 252-day Sharpe (annualised)")
    return style_fig(fig, height=300)


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
        x=df["contribution_to_return"], y=df["symbol"], orientation="h", marker_color=colors
    )
    fig.update_xaxes(tickformat=".1%")
    label = _STRATEGY_LABELS.get(strategy, strategy)
    fig.update_layout(title=f"{label} — return contribution by asset")
    return style_fig(fig, height=340)


def regime_sharpe_chart(regime: pd.DataFrame) -> go.Figure:
    """Grouped bars: annualised Sharpe by market volatility regime, one bar per strategy."""
    order = ["Low", "Medium", "High"]
    fig = go.Figure()
    for idx, strategy in enumerate(sorted(regime["strategy"].unique())):
        grp = regime[regime["strategy"] == strategy].set_index("regime").reindex(order)
        fig.add_bar(
            x=order,
            y=grp["ann_sharpe"],
            name=_STRATEGY_LABELS.get(strategy, strategy),
            marker_color=_strategy_line(strategy, idx)["color"],
        )
    fig.update_layout(title="Annualised Sharpe by market volatility regime", barmode="group")
    return style_fig(fig, height=340)


def ml_gate_chart(gate: pd.DataFrame) -> go.Figure:
    """The ML gate over time: weight the forecast earns in mvo_ml (0 = no out-of-sample edge)."""
    fig = go.Figure()
    fig.add_scatter(
        x=gate["date"],
        y=gate["forecast_weight"],
        name="forecast weight (λ)",
        line=dict(color=PALETTE["series"][3]),
    )
    fig.add_scatter(
        x=gate["date"],
        y=gate["forecast_skill"],
        name="forecast skill",
        line=dict(color=PALETTE["muted"], dash="dash"),
    )
    fig.update_layout(title="ML gate — forecast weight & skill over time")
    fig.update_yaxes(rangemode="tozero")
    return style_fig(fig, height=300)


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
