"""Plotly chart builders — all styling routed through theme.style_fig."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from dashboard.theme import (
    HEIGHT_DEFAULT,
    HEIGHT_MEDIUM,
    HEIGHT_TALL,
    PALETTE,
    SERIES_ALT,
    SERIES_PRICE,
    SERIES_RETURN,
    SERIES_VOL,
    SERIES_YIELD,
    asset_class_color,
    style_fig,
)

from mmi.ml.skill_gate import skill_verdict

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
    # All assets are USD-denominated (equities, GLD, BTC, and the USD-quoted FX pairs), so show the
    # axis as $ with thousands separators; 2dp keeps low-priced assets (FX ≈ 1.2) readable.
    # hoverformat matches so the hover tooltip reads the same as the ticks ($171.80, not 171.7959).
    fig.update_yaxes(tickformat="$,.2f", hoverformat="$,.2f")
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
    # vol_20d is annualised; render axis + hover as a percentage.
    fig.update_yaxes(tickformat=".1%", hoverformat=".1%")
    if not df.empty and "vol_20d" in df.columns:
        _guard_yrange(fig, df["vol_20d"])
    return style_fig(fig, height=HEIGHT_DEFAULT)


# ---------------------------------------------------------------------------
# Markets tab — cross-asset view (leaderboard · rebased performance · correlation)
# ---------------------------------------------------------------------------

#: Annualisation factor for daily-return volatility (trading days per year).
_TRADING_DAYS: int = 252
#: Minimum overlapping observations before a correlation matrix is considered stable.
_CORR_MIN_OBS: int = 30
#: Shown instead of a misleading matrix when the window holds too few observations.
CORR_TOO_SHORT: str = "Range too short for a stable correlation — widen the date range."


def cross_asset_leaderboard(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per-asset period stats over the supplied (already windowed) long frame.

    Input is the ``[symbol, asset_class, date, close, daily_return]`` long frame from
    ``data.all_assets_daily(start)``. For each symbol, over the window:
      * ``period_return`` = ``close.iloc[-1] / close.iloc[0] - 1`` (close-to-close over the window)
      * ``ann_vol``       = ``daily_return.std() * sqrt(252)`` (annualised daily-return vol)
    Returns ``[symbol, asset_class, period_return, ann_vol]`` sorted by ``period_return`` desc.
    Pure + unit-tested — the leaderboard cards read straight off this frame.
    """
    cols = ["symbol", "asset_class", "period_return", "ann_vol"]
    if long_df.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict] = []
    for symbol, grp in long_df.groupby("symbol", sort=False):
        g = grp.sort_values("date")
        closes = g["close"].dropna()
        if len(closes) < 2 or closes.iloc[0] == 0:
            continue  # need at least two prices for a period return
        period_return = float(closes.iloc[-1] / closes.iloc[0] - 1)
        ann_vol = float(g["daily_return"].std(ddof=1) * math.sqrt(_TRADING_DAYS))
        asset_class = str(g["asset_class"].iloc[0])
        rows.append(
            {
                "symbol": str(symbol),
                "asset_class": asset_class,
                "period_return": period_return,
                "ann_vol": ann_vol,
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("period_return", ascending=False).reset_index(drop=True)


def rebased_performance(long_df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative-return path per symbol, rebased to 0% at the window start.

    For each symbol: ``perf = (1 + daily_return).cumprod() - 1`` over the (already windowed) rows,
    with the window's FIRST daily_return treated as 0 so every line starts at exactly 0% on the
    range's first date. Returns the long frame ``[symbol, asset_class, date, perf]`` ordered by
    ``symbol, date``. Pure + unit-tested.
    """
    cols = ["symbol", "asset_class", "date", "perf"]
    if long_df.empty:
        return pd.DataFrame(columns=cols)
    out_parts: list[pd.DataFrame] = []
    for _symbol, grp in long_df.groupby("symbol", sort=False):
        g = grp.sort_values("date").copy()
        # The window's first row has no in-window return — pin it to 0 so the line starts at 0%.
        r = g["daily_return"].fillna(0.0).to_numpy(dtype=float).copy()
        if len(r):
            r[0] = 0.0
        g["perf"] = (1.0 + r).cumprod() - 1.0
        out_parts.append(g[cols])
    return pd.concat(out_parts, ignore_index=True)


def correlation_matrix(long_df: pd.DataFrame) -> pd.DataFrame | None:
    """Pairwise Pearson correlation of daily returns over the window, or ``None`` if too short.

    Pivots the long frame to ``date × symbol`` of ``daily_return`` and returns ``.corr()``. The
    **min-obs guard**: if fewer than ``_CORR_MIN_OBS`` (~30) rows have at least two non-null
    symbol returns to correlate, returns ``None`` (the caller shows ``CORR_TOO_SHORT``) rather
    than a misleading matrix from a handful of points. Pure + unit-tested.
    """
    if long_df.empty:
        return None
    wide = long_df.pivot_table(index="date", columns="symbol", values="daily_return")
    # Overlapping observations = rows where at least two symbols have a (non-null) return to pair.
    overlap = int((wide.notna().sum(axis=1) >= 2).sum())
    if overlap < _CORR_MIN_OBS or wide.shape[1] < 2:
        return None
    return wide.corr()


def correlation_takeaway(corr: pd.DataFrame) -> str:
    """One-line, data-honest takeaway under the heatmap (highest + lowest off-diagonal pair)."""
    if corr is None or corr.empty or corr.shape[0] < 2:
        return ""
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append((str(cols[i]), str(cols[j]), float(v)))
    if not pairs:
        return ""
    hi = max(pairs, key=lambda p: p[2])
    lo = min(pairs, key=lambda p: p[2])
    return (
        f"Most correlated: {hi[0]}–{hi[1]} ({hi[2]:+.2f}); "
        f"best diversifier: {lo[0]}–{lo[1]} ({lo[2]:+.2f}). "
        "Equities tend to cluster; bonds and gold/BTC usually diversify."
    )


def leaderboard_return_color(period_return: float) -> str:
    """Green for a positive period return, red for negative (for assets, up = good)."""
    return PALETTE["up"] if period_return >= 0 else PALETTE["down"]


def rebased_performance_chart(perf_long: pd.DataFrame, height: int = HEIGHT_TALL) -> go.Figure:
    """One class-coloured line per symbol over the window, each rebased to 0% at the start.

    The legend shows each symbol with its final % so the chart reads without hovering. Line
    colour comes from the asset-class colour map (``theme.asset_class_color``)."""
    fig = go.Figure()
    if not perf_long.empty:
        for symbol, grp in perf_long.groupby("symbol", sort=False):
            g = grp.sort_values("date")
            asset_class = str(g["asset_class"].iloc[0]) if "asset_class" in g else ""
            final = float(g["perf"].iloc[-1]) if not g["perf"].empty else 0.0
            fig.add_scatter(
                x=g["date"],
                y=g["perf"],
                name=f"{symbol}  {final * 100:+.1f}%",
                line=dict(color=asset_class_color(asset_class)),
            )
    fig.add_hline(y=0, line_color=PALETTE["muted"], line_dash="dot")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(
        title=dict(text="Cross-asset performance — rebased to 0% at window start", font=_TITLE_FONT)
    )
    _apply_axis_fonts(fig)
    n = perf_long["symbol"].nunique() if not perf_long.empty else 0
    _overflow_legend(fig, n)
    return style_fig(fig, height=height)


def correlation_heatmap(corr: pd.DataFrame, height: int = HEIGHT_TALL) -> go.Figure:
    """Annotated Pearson-correlation heatmap on a diverging RdBu scale fixed to −1..1."""
    symbols = list(corr.columns)
    z = corr.to_numpy()
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=symbols,
            y=symbols,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,  # red = positive correlation, blue = negative (diversifying)
            colorbar=dict(title="ρ", tickfont=_AXIS_FONT),
            text=[[f"{v:+.2f}" if pd.notna(v) else "" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=11, color=PALETTE["text"]),
            hovertemplate="%{y} · %{x}: %{z:+.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Cross-asset correlation — daily returns over the window", font=_TITLE_FONT)
    )
    fig.update_yaxes(autorange="reversed")  # diagonal runs top-left → bottom-right
    _apply_axis_fonts(fig)
    return style_fig(fig, height=height)


# ---------------------------------------------------------------------------
# Macro tab
# ---------------------------------------------------------------------------


def macro_chart(
    df: pd.DataFrame, label: str, units: str = "", height: int | None = None
) -> go.Figure:
    fig = go.Figure()
    # A sparse series — quarterly data, or any series viewed over a short date range — reads as a
    # flat/near-empty line, so show markers when there are few points to keep the observations
    # visible. Frequent (daily/monthly) series stay clean lines.
    mode = "lines+markers" if len(df) <= 40 else "lines"
    # Macro values are stored in their native scale (percent series already ×100, e.g. 4.3; indices
    # and $ series raw), so we do NOT use a Plotly "%" tickformat (it would render 4.3 as 430%).
    # The hover shows the value at 2dp with the unit appended, so "4.30 %" / "120.40 index".
    unit_suffix = f" {units}" if units else ""
    fig.add_scatter(
        x=df["date"],
        y=df["value"],
        name=label,
        mode=mode,
        line=dict(color=PALETTE["accent"]),
        hovertemplate=f"%{{x|%Y-%m-%d}}: %{{y:,.2f}}{unit_suffix}<extra></extra>",
    )
    title = f"{label} · {units}" if units else label
    fig.update_layout(title=dict(text=title, font=_TITLE_FONT))
    _apply_axis_fonts(fig)
    return style_fig(fig, height=height or HEIGHT_DEFAULT)


def yield_curve_chart(df: pd.DataFrame) -> go.Figure:
    """Yield-curve spread — canonical 10Y−3M when available, else the 10Y−2Y proxy.

    The recession-risk panel uses the 10Y−3M spread (NY Fed / Estrella-Mishkin canonical), so this
    inversion chart mirrors it, falling back to 10Y−2Y only when the 3-month series is unavailable.
    Inversion belt is below the zero line.
    """
    use_3m = "yield_curve_10y_3m" in df.columns and df["yield_curve_10y_3m"].notna().any()
    col = "yield_curve_10y_3m" if use_3m else "yield_curve_10y_2y"
    label = "10Y − 3M" if use_3m else "10Y − 2Y"

    fig = go.Figure()
    fig.add_scatter(
        x=df["date"],
        y=df[col],
        name=f"{label} spread",
        line=dict(color=SERIES_YIELD),
        hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f} pp<extra></extra>",
    )
    fig.add_hline(y=0, line_color=PALETTE["down"], line_dash="dot")
    fig.update_layout(
        title=dict(
            text=f"Yield-curve spread ({label}) — inversion below 0",
            font=_TITLE_FONT,
        ),
    )
    _apply_axis_fonts(fig)
    # Y-range guard: keep the zero-line visible with symmetric padding
    if not df.empty and col in df.columns:
        series = df[col].dropna()
        if not series.empty:
            lo, hi = float(series.min()), float(series.max())
            span = max(abs(lo), abs(hi), 0.5)
            fig.update_yaxes(range=[-span * 1.15, span * 1.15])
    return style_fig(fig, height=HEIGHT_DEFAULT)


# ---------------------------------------------------------------------------
# Macro tab — recession-risk panel (E3)
# ---------------------------------------------------------------------------

#: Scope/caveat caption baked in per Contract E.
#: The panel is macro CONTEXT only — it must NOT be read as a return or price forecast.
_RECESSION_RISK_CAVEATS: str = (
    "**Context only — not a forecast.**  "
    "Yield-curve recession-risk models (Estrella & Mishkin 1998) show AUC ~0.85–0.89 at a "
    "12-month horizon (San Francisco Fed 2018).  "
    "**Caveat 1 — term-premium critique:** a theoretically-motivated term-premium adjustment "
    "(Bauer & Mertens 2018) actually *lowers* predictive AUC; the unadjusted spread is used here.  "
    "**Caveat 2 — 2022–23 false positive:** the 2022–23 yield-curve inversion produced a "
    "sharply elevated recession probability, yet no NBER recession was declared through mid-2026. "
    "This is a documented, live out-of-sample failure — the model was not re-tuned to pass. "
    "Use this panel for macro regime awareness, not as a recession signal."
)

#: Model label for the chart title, keyed by the mart's 'model' column value.
_RECESSION_MODEL_LABELS: dict[str, str] = {
    "10y_3m": "10Y−3M spread (canonical Estrella-Mishkin)",
    "10y_2y_proxy": "10Y−2Y spread (proxy — 3M series unavailable)",
}


def recession_risk_chart(df: pd.DataFrame) -> go.Figure:
    """Recession probability over time from the Estrella-Mishkin probit.

    Uses the ``model`` column from ``fct_recession_risk`` to label whether the chart
    is using the canonical 10Y−3M spread or the 10Y−2Y proxy.  Colours come from
    named PALETTE tokens only — no inline hex.  A reference line at 0.50 marks the
    conventional "high-probability" threshold.
    """
    fig = go.Figure()

    # Determine model label from the mart's model column (use the first non-null value).
    model_key: str = "10y_3m"  # default
    if not df.empty and "model" in df.columns:
        first_model = df["model"].dropna().iloc[0] if not df["model"].dropna().empty else model_key
        model_key = str(first_model)
    model_label = _RECESSION_MODEL_LABELS.get(model_key, model_key)

    # Recession probability filled area — amber (SERIES_VOL) signals caution without falsely
    # implying "recession confirmed" (which would be PALETTE["down"]/red).
    fig.add_scatter(
        x=df["date"] if not df.empty else [],
        y=df["recession_prob"] if not df.empty else [],
        name="Recession probability",
        fill="tozeroy",
        line=dict(color=SERIES_VOL),
        fillcolor="rgba(255,180,84,0.18)",  # SERIES_VOL (#ffb454) at ~18% opacity
    )

    # Overlay the underlying yield-curve spread on a second y-axis for context.
    if not df.empty and "spread_10y_3m" in df.columns and df["spread_10y_3m"].notna().any():
        fig.add_scatter(
            x=df["date"],
            y=df["spread_10y_3m"],
            name="Yield-curve spread (pp)",
            line=dict(color=SERIES_YIELD, dash="dot"),
            yaxis="y2",
        )

    # 50% reference line — conventional "elevated risk" threshold.
    fig.add_hline(
        y=0.50,
        line_color=PALETTE["down"],
        line_dash="dot",
        annotation_text="50% threshold",
        annotation_font_color=PALETTE["down"],
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(
            text=f"Recession-risk probability — {model_label}",
            font=_TITLE_FONT,
        ),
        yaxis=dict(
            title="Recession probability",
            tickformat=".0%",
            range=[0, 1.05],
        ),
        yaxis2=dict(
            title="Spread (pp)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    _apply_axis_fonts(fig)
    return style_fig(fig, height=HEIGHT_DEFAULT)


def recession_risk_caption(is_sample: bool | None) -> str:
    """Source caption for the recession-risk panel.

    The recession probability is *derived* from FRED yield series when data is live,
    and from synthetic seed yields when in sample mode.  Routes through ``is_sample``
    tri-state exactly as ``macro_source_caption`` does — never hardcodes 'Source: FRED'.
    """
    if is_sample is False:
        return (
            "Recession probability derived from FRED DGS10 and DGS3MO (or DGS2) yield series · "
            "Estrella & Mishkin (1998) probit model · "
            "https://fred.stlouisfed.org/"
        )
    if is_sample is True:
        return (
            "⚠️ Recession probability derived from synthetic seed yields — "
            "not from FRED (live data uses FRED DGS10 / DGS3MO)."
        )
    return ""  # mixed / unknown provenance → make no source claim


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


def regime_chart(df: pd.DataFrame, symbol: str, height: int = HEIGHT_MEDIUM) -> go.Figure:
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
    return style_fig(fig, height=height)


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
# ML tab — honest vol-skill builders (B7)
# ---------------------------------------------------------------------------

#: Scope caption for the ML forecast tab.
ML_SCOPE_CAPTION: str = (
    "Models: cross-asset return forecaster (gradient boosting) + realised-volatility forecaster "
    "(HAR/EWMA baseline check)"
)


def vol_skill_r2_chart(
    metrics: pd.DataFrame, symbol: str = "SPY", height: int = HEIGHT_MEDIUM
) -> go.Figure:
    """Grouped bars: OOS R² for the volatility model vs the persistence/EWMA baseline.

    Baseline R² is always 0 by construction (persistence = the null model), so this
    chart shows whether the vol model explains any variance beyond naive persistence.
    The bar colours use named PALETTE tokens — no inline hex.
    """
    m = metrics[(metrics["model"] == "rv_har") & (metrics["symbol"] == symbol)].set_index("metric")[
        "value"
    ]
    oos_r2 = float(m.get("oos_r2", 0.0) or 0.0)
    # Persistence (the baseline) has R²=0 by definition; we show it explicitly for context.
    baseline_r2: float = 0.0

    fig = go.Figure()
    fig.add_bar(
        x=["Vol model", "Persistence / EWMA baseline"],
        y=[oos_r2, baseline_r2],
        marker_color=[PALETTE["accent"], PALETTE["muted"]],
        name="OOS R²",
    )
    fig.add_hline(
        y=0.10,
        line_color=PALETTE["up"],
        line_dash="dot",
        annotation_text="skill gate (R²≥0.10)",
        annotation_font_color=PALETTE["up"],
        annotation_position="top right",
    )
    fig.update_yaxes(title_text="Out-of-sample R²")
    fig.update_layout(
        title=dict(
            text=f"{symbol} — HAR vol-model OOS R² vs persistence baseline",
            font=_TITLE_FONT,
        ),
        showlegend=False,
    )
    _apply_axis_fonts(fig)
    return style_fig(fig, height=height)


def vol_skill_qlike_chart(
    metrics: pd.DataFrame, symbol: str = "SPY", height: int = HEIGHT_MEDIUM
) -> go.Figure:
    """Grouped bars: model QLIKE vs baseline QLIKE, with the skill-ratio annotated.

    Lower QLIKE is better (it is a proper scoring rule for volatility forecasts).
    The skill ratio (model / baseline) is annotated; < 0.99 is the go-live threshold.
    """
    m = metrics[(metrics["model"] == "rv_har") & (metrics["symbol"] == symbol)].set_index("metric")[
        "value"
    ]
    model_qlike = float(m.get("qlike", float("nan")) or float("nan"))
    baseline_qlike = float(m.get("baseline_qlike", float("nan")) or float("nan"))
    skill_ratio = float(m.get("qlike_skill_ratio", float("nan")) or float("nan"))

    ratio_label = f"{skill_ratio:.3f}" if not math.isnan(skill_ratio) else "n/a"

    fig = go.Figure()
    fig.add_bar(
        x=["Vol model", "Persistence / EWMA baseline"],
        y=[
            model_qlike if not math.isnan(model_qlike) else 0.0,
            baseline_qlike if not math.isnan(baseline_qlike) else 0.0,
        ],
        marker_color=[SERIES_VOL, PALETTE["muted"]],
        name="QLIKE",
    )
    # Annotate the skill ratio on the model bar.
    fig.add_annotation(
        x="Vol model",
        y=model_qlike if not math.isnan(model_qlike) else 0.0,
        text=f"skill ratio: {ratio_label}",
        showarrow=False,
        yshift=12,
        font=dict(color=PALETTE["text"], size=11),
    )
    fig.add_hline(
        y=baseline_qlike * 0.99 if not math.isnan(baseline_qlike) else 0.0,
        line_color=PALETTE["up"],
        line_dash="dot",
        annotation_text="gate: ratio < 0.99",
        annotation_font_color=PALETTE["up"],
        annotation_position="top right",
    )
    fig.update_yaxes(title_text="QLIKE (lower = better)")
    fig.update_layout(
        title=dict(
            text=f"{symbol} — vol-model QLIKE vs baseline (skill ratio annotated)",
            font=_TITLE_FONT,
        ),
        showlegend=False,
    )
    _apply_axis_fonts(fig)
    return style_fig(fig, height=height)


def vol_skill_verdict_text(metrics: pd.DataFrame, symbol: str = "SPY") -> str:
    """One honest verdict string sourced ONLY from skill_verdict().

    Returns 'beats baseline OOS' language ONLY when cleared=True; otherwise
    returns an honest 'no demonstrated out-of-sample edge — baseline-only' message.
    """
    verdict = skill_verdict(metrics, symbol=symbol)
    if verdict["cleared"]:
        r2 = verdict["oos_r2"]
        ratio = verdict["qlike_skill_ratio"]
        folds_passed = verdict["folds_passed"]
        n_folds = verdict["n_folds"]
        return (
            f"{symbol} volatility model beats baseline OOS "
            f"(OOS R²={r2:.3f} ≥ 0.10; QLIKE skill ratio={ratio:.3f} < 0.99; "
            f"{folds_passed}/{n_folds} folds passed)."
        )
    reasons = "; ".join(verdict["reasons"]) if verdict["reasons"] else "metrics not yet available"
    return f"Volatility model: no demonstrated out-of-sample edge — baseline-only. {reasons}."


#: Honest framing for the locked-holdout readout — an extra OOS look, never a gate.
HOLDOUT_CAPTION: str = (
    "Locked holdout (last ~1yr, never used in CV — an extra out-of-sample readout, not gated)"
)


def holdout_readout(
    metrics: pd.DataFrame,
    model: str | None = None,
    symbol: str = "SPY",
    *,
    exclude_model: str | None = None,
) -> dict | None:
    """The locked-holdout metrics for a model/``symbol``, or ``None`` when none are present.

    Reads the ``holdout_*`` rows PR #17 added to ``model_metrics``:
      * vol (``rv_har``):  ``holdout_oos_r2``, ``holdout_qlike_skill_ratio``, ``holdout_n_obs``
      * direction:  ``holdout_dir_acc``, ``holdout_baseline_dir_acc``, ``holdout_n_obs``
    Pass ``model`` to match a model exactly (the vol headline → ``model="rv_har"``), OR
    ``exclude_model`` to take the OTHER model (the direction secondary → ``exclude_model="rv_har"``,
    mirroring ``direction_skill_chart``'s robust "not the vol model" filter).
    Returns ``None`` (render nothing / "pending") when the frame is empty, lacks the expected
    columns, or carries no ``holdout_*`` row for this model/symbol — the holdout is SKIPPED on
    small-data (CI/sample) and absent from pre-re-run snapshots, so absence must degrade
    gracefully. Otherwise returns a dict of the present ``holdout_*`` metric → float value
    (only finite values are kept). Pure + unit-tested."""
    needed = {"model", "symbol", "metric", "value"}
    if metrics.empty or not needed <= set(metrics.columns):
        return None
    rows = metrics[metrics["symbol"] == symbol]
    if model is not None:
        rows = rows[rows["model"] == model]
    if exclude_model is not None:
        rows = rows[rows["model"] != exclude_model]
    if rows.empty:
        return None
    s = rows.set_index("metric")["value"]
    out: dict[str, float] = {}
    for key in s.index:
        if not str(key).startswith("holdout_"):
            continue
        val = s[key]
        if val is None or pd.isna(val):
            continue
        fval = float(val)
        if math.isfinite(fval):
            out[str(key)] = fval
    return out or None


def vol_forecast_value(fc: pd.DataFrame, symbol: str = "SPY") -> float | None:
    """Predicted next-week annualised realised vol for ``symbol`` from the rv_har forecast.

    Filters the ``ml_forecast`` frame on BOTH ``model == 'rv_har'`` AND ``symbol`` — never the
    model alone — so that if the ML run ever covers multiple symbols (a future config override),
    the positional ``.iloc[0]`` can't surface another asset's forecast in the SPY headline.
    Returns ``None`` (honest empty state, no IndexError) when there is no matching row, the
    frame lacks the expected columns, or the matched value is null/non-finite.
    """
    needed = {"model", "symbol", "predicted_next_return"}
    if fc.empty or not needed <= set(fc.columns):
        return None
    rows = fc[(fc["model"] == "rv_har") & (fc["symbol"] == symbol)]
    if rows.empty:
        return None
    value = rows["predicted_next_return"].iloc[0]
    # A null/non-finite forecast must surface as the honest empty caption ("No SPY volatility
    # forecast available yet."), never as a "looks-valid-but-isn't" "nan %" headline. Returning
    # None here keeps the caller's `is not None` branch falsy. pd.isna also guards against a
    # None/NULL object-dtype value raising TypeError in float(); math.isfinite additionally
    # rejects ±inf (which pd.isna treats as non-null).
    if value is None or pd.isna(value):
        return None
    forecast = float(value)
    if not math.isfinite(forecast):
        return None
    return forecast * math.sqrt(252)


def direction_skill_chart(metrics: pd.DataFrame, symbol: str = "SPY") -> go.Figure:
    """Paired bars: next-day direction model MAE and directional accuracy vs baseline.

    This is EXPLICITLY LABELLED as an honest secondary — there is no demonstrated
    short-horizon edge for the direction model (Contract E, demoted status).
    Both PALETTE named tokens are used; no inline hex.
    """
    # Direction rows carry model='random_forest' (or similar); filter broadly on !='rv_har'
    # so this chart works with whatever direction model name is present.
    dir_rows = metrics[(metrics["model"] != "rv_har") & (metrics["symbol"] == symbol)]
    m = dir_rows.set_index("metric")["value"] if not dir_rows.empty else pd.Series(dtype=float)

    mae_model = float(m.get("mae", float("nan")) or float("nan"))
    mae_base = float(m.get("mae_baseline", float("nan")) or float("nan"))
    dir_acc = float(m.get("dir_acc", float("nan")) or float("nan"))
    dir_acc_base = float(m.get("baseline_dir_acc", float("nan")) or float("nan"))

    def _safe(v: float) -> float:
        return 0.0 if math.isnan(v) else v

    fig = go.Figure()
    # MAE bars (lower is better — use SERIES_RISK for model to signal caution)
    fig.add_bar(
        x=["Model MAE", "Baseline MAE"],
        y=[_safe(mae_model), _safe(mae_base)],
        marker_color=[SERIES_RETURN, PALETTE["muted"]],
        name="MAE",
        offsetgroup=0,
    )
    # Dir-acc bars (higher is better)
    fig.add_bar(
        x=["Model dir-acc", "Baseline dir-acc"],
        y=[_safe(dir_acc), _safe(dir_acc_base)],
        marker_color=[SERIES_PRICE, PALETTE["muted"]],
        name="Dir accuracy",
        offsetgroup=1,
    )
    fig.update_layout(
        title=dict(
            text=(
                f"{symbol} — next-day direction model vs baseline "
                "(honest secondary — no demonstrated short-horizon edge)"
            ),
            font=_TITLE_FONT,
        ),
        barmode="group",
    )
    _apply_axis_fonts(fig)
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


def rebase_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Rebase each strategy's ``cumulative_return`` to 0% at its first (visible) row, so the chart
    reads as 'return since the start of the selected range' (Google-Finance style).

    Exact: for a windowed slice ``(1 + cum_t) / (1 + cum_first) - 1`` equals the compounded return
    over the visible rows (the mart's ``cumulative_return`` is inception-based, so a sub-range would
    otherwise start mid-history at e.g. +150%). Shape-unchanged at 'Max' (first row ~ inception).
    Pure + unit-tested."""
    if df.empty or "cumulative_return" not in df.columns:
        return df
    out = df.sort_values(["strategy", "date"]).copy()
    base = out.groupby("strategy")["cumulative_return"].transform("first")
    out["cumulative_return"] = (1 + out["cumulative_return"]) / (1 + base) - 1
    return out


def portfolio_cumulative_chart(df: pd.DataFrame, height: int = HEIGHT_TALL) -> go.Figure:
    fig = _by_strategy(rebase_cumulative(df), "cumulative_return")
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
    return style_fig(fig, height=height)


def portfolio_drawdown_chart(df: pd.DataFrame, height: int = HEIGHT_MEDIUM) -> go.Figure:
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
    return style_fig(fig, height=height)


def portfolio_sharpe_chart(df: pd.DataFrame, height: int = HEIGHT_MEDIUM) -> go.Figure:
    fig = _by_strategy(df, "rolling_sharpe_252")
    fig.update_layout(
        title=dict(text="Rolling 252-day Sharpe (annualised)", font=_TITLE_FONT),
    )
    _apply_axis_fonts(fig)
    n = df["strategy"].nunique() if not df.empty else 0
    _overflow_legend(fig, n)
    return style_fig(fig, height=height)


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
