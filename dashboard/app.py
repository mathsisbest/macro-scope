"""Markets & Macro Intelligence — Streamlit dashboard (BI layer).

Run: `make dashboard` / `make demo`, or `streamlit run dashboard/app.py` directly — this file
puts the repo root on sys.path so `from dashboard import ...` resolves everywhere (local and
Streamlit Community Cloud, which otherwise only has this file's own dir on the path).
Reads the dbt marts from DuckDB; everything visual is defined in code.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud runs this file with only its own directory on sys.path (not the
# repo root), so the repo-root `dashboard` package isn't importable. Put the repo root first.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dashboard.snapshot_boot import configure_dashboard_env  # noqa: E402

# Make config visible to pydantic-settings (which reads env vars) BEFORE the settings singleton
# is built below. Streamlit Community Cloud exposes secrets via st.secrets and does not reliably
# promote them to env vars, so bridge any scalar secret into the environment first (real env vars
# win via setdefault).
with contextlib.suppress(Exception):  # no secrets.toml in local dev — that's fine
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))

# Pin the committed-snapshot dir to this checkout and default to snapshot mode when there's no
# live DB — makes the public app zero-config and correct even on a non-editable package install.
configure_dashboard_env(os.environ, _REPO_ROOT)

from dashboard import data  # noqa: E402
from dashboard.components import charts  # noqa: E402
from dashboard.components.kpi import metric_row  # noqa: E402
from dashboard.theme import PLOTLY_CONFIG, inject_css  # noqa: E402

from mmi.settings import settings  # noqa: E402
from mmi.utils.db import connect  # noqa: E402

# --------------------------------------------------------------------------- page config
_FAVICON = Path(__file__).resolve().parent / "assets" / "favicon.png"
st.set_page_config(
    page_title="Macro Scope",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "📈",
    layout="wide",
)
inject_css()


def _chart(fig, **kwargs):
    """Thin wrapper so every chart gets the mobile-safe config (no scroll-zoom, no modebar)."""
    kwargs.setdefault("config", PLOTLY_CONFIG)
    st.plotly_chart(fig, use_container_width=True, **kwargs)


# --------------------------------------------------------------------------- hero / header
st.title("📈 Macro Scope")
st.subheader("Markets & Macro Intelligence")
st.caption(
    "Live markets + macro · **ingest → dbt → ML → GenAI → BI** · all free tiers · "
    "walk-forward backtesting · no secrets required in the public app"
)

# --------------------------------------------------------------------------- methodology expander
with st.expander("About & methodology", expanded=False):
    st.markdown(
        """
**Pipeline**

`mmi ingest` → `dbt build` → `mmi ml` → `mmi ai` → Streamlit BI

Each stage is open-source and runs on free-tier infrastructure (Yahoo Finance unofficial API,
FRED, World Bank, DuckDB, scikit-learn, a local or serverless LLM).

**Data sources**

- **Yahoo Finance (unofficial)** — equities, ETFs, FX and BTC (BTC-USD) daily OHLCV.
  Unofficial API; not endorsed by Yahoo Finance.
- **FRED — Federal Reserve Bank of St. Louis** — macro series (CPI, unemployment,
  Fed Funds rate, yield curve). [fred.stlouisfed.org](https://fred.stlouisfed.org/)
- **World Bank** — additional macro indicators.
  [data.worldbank.org](https://data.worldbank.org/)

**ML headline target**

The certified ML model forecasts **next-week (5-trading-day) realised volatility for SPY** using
a HAR-style cascade (1d / 5d / 22d Garman-Klass vol + yield-curve macro features).
Walk-forward `TimeSeriesSplit(5)` — no lookahead leakage.
The go-live skill gate is `OOS R² ≥ 0.10 AND QLIKE-ratio < 0.99 AND ≥ 3/5 folds pass`.
If the gate is not cleared the dashboard shows the honest "no demonstrated out-of-sample edge"
state — the model is never re-tuned to pass.

**Bond-return honesty note (TLT / TIP)**

Bond-return predictability is well-documented **in-sample**: Fama-Bliss forward-rate regressions
achieve ~15% R², and Cochrane-Piazzesi factors reach up to 0.44.  However, the evidence is
**fragile out-of-sample** — Thornton & Valente (2012), Hodrick & Tomunen (2021), and Bauer &
Hamilton (2018) all find that the in-sample gains largely disappear once accounting for
data-snooping, statistical uncertainty, and real-time revision.

**This is why mmi weights TLT and TIP by risk** (inverse-vol / risk-parity / MVO), **not by a
return forecast**: the data cannot honestly support a forward-rate predictor, so we rely only on
the diversification benefit of bonds within a risk-constrained portfolio.

**Not investment advice**

Nothing here constitutes financial, investment, or trading advice.
All backtests are historical and do not guarantee future results.
Use at your own risk.
        """.strip()
    )

if not data.db_exists():
    st.warning(
        "No database yet. Run `make demo` (or `mmi seed`) to populate sample data, then reload."
    )
    st.stop()

# --------------------------------------------------------------------------- data provenance
# Honest "data as of <date> · sample/live" badge. Both signals come from the marts, so they are
# correct in BOTH live and snapshot (public Parquet) mode — raw.pipeline_runs isn't snapshotted.
as_of = data.data_as_of()
is_sample = data.is_sample_data()
provenance = [f"📅 Data as of **{as_of}**"] if as_of else []
if is_sample is True:
    provenance.append("🧪 sample data (synthetic — run `mmi ingest` for live)")
elif is_sample is False:
    provenance.append("🟢 live data")
elif as_of:
    provenance.append("⚠️ mixed/unrecorded data provenance")
if provenance:
    st.caption(" · ".join(provenance))


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    with st.expander("⚙️ Pipeline health", expanded=False):
        runs = data.pipeline_runs()
        if not runs.empty:
            st.dataframe(runs, hide_index=True)
        elif is_sample is True:
            st.caption("Sample data seeded (synthetic; no live ingestion runs).")
        elif is_sample is False:
            st.caption("Live data from a committed snapshot (no in-app ingestion log).")
        elif as_of:
            st.caption("Mixed or unrecorded data provenance.")
        else:
            st.caption("No data yet — run `make demo` or `mmi ingest`.")
    st.divider()
    st.caption(f"`{settings.storage_label()}`")
    st.caption(f"LLM provider · `{settings.llm_provider}`")


# --------------------------------------------------------------------------- KPI row
# Headline figures always show the LATEST value (unaffected by the date-range selector below).
kpis: list[dict] = []
btc = data.asset_daily("BTC")
if not btc.empty:
    br = btc["daily_return"].iloc[-1]
    kpis.append(
        {
            "label": "BTC close",
            "value": f"${btc['close'].iloc[-1]:,.0f}",
            "delta": f"{(br or 0) * 100:+.2f}%",
        }
    )

spy = data.asset_daily("SPY")
if not spy.empty:
    r = spy["daily_return"].iloc[-1]
    kpis.append(
        {
            "label": "SPY close",
            "value": f"${spy['close'].iloc[-1]:,.2f}",
            "delta": f"{(r or 0) * 100:+.2f}%",
        }
    )

reg = data.regimes("SPY")
if not reg.empty:
    kpis.append({"label": "SPY vol regime", "value": str(reg["regime"].iloc[-1])})

mm = data.market_macro()
# Prefer the canonical 10Y−3M spread (NY Fed / Estrella-Mishkin — the inversion investors watch
# for recession risk, and what the recession-risk panel uses); fall back to 10Y−2Y when the 3M
# series is unavailable (e.g. a snapshot taken before the 10Y−3M column existed).
if not mm.empty and "yield_curve_10y_3m" in mm.columns and mm["yield_curve_10y_3m"].notna().any():
    spread = mm["yield_curve_10y_3m"].dropna().iloc[-1]
    kpis.append({"label": "10Y−3M spread", "value": f"{spread:+.2f} pp"})
elif not mm.empty and mm["yield_curve_10y_2y"].notna().any():
    spread = mm["yield_curve_10y_2y"].dropna().iloc[-1]
    kpis.append({"label": "10Y−2Y spread", "value": f"{spread:+.2f} pp"})

if kpis:
    metric_row(kpis)

st.divider()

# --------------------------------------------------------------------------- global date range
# One Google-Finance-style selector that filters EVERY time-series chart across all tabs. The
# aggregate stat panels (bootstrap Sharpe CIs, attribution, BTC effect) stay full-window — they
# are window-level statistics, not per-row series.
_range = st.segmented_control(
    "Date range",
    data.RANGE_PRESETS,
    default="5Y",
    key="chart_range",
    help="Filters every time-series chart. Portfolio bootstrap stats stay full-window.",
)
rng_start = data.range_start(_range, as_of)

# --------------------------------------------------------------------------- tabs
# Human labels for the Phase-D backtest windows (the selector in the Portfolio tab).
_WINDOW_LABELS = {
    "ex_btc_2002": "~2004–present · ex-BTC",
    "ex_btc_2015": "2015–present · ex-BTC (BTC era)",
    "inc_btc_2015": "2015–present · incl. BTC",
}

tab_mkt, tab_macro, tab_ml, tab_ai, tab_portfolio = st.tabs(
    ["Markets", "Macro", "ML forecast", "AI brief", "Portfolio"]
)

with tab_mkt:
    # Cross-asset view spanning every class — equities, bonds, commodities, FX AND crypto (BTC has
    # full daily history in fct_asset_daily, so it joins the cross-asset stats like any other
    # asset). Four layers, overview → relationships → drill-down, all governed by the global range:
    #   1. leaderboard (over-the-window return + ann. vol, one card per asset)
    #   2. rebased performance (every line starts at 0% on the range's first date)
    #   3. correlation heatmap (pairwise Pearson of daily returns over the window)
    #   4. per-asset drill-down (the original single-asset price + vol view, windowed)
    # Headline "latest" values stay range-independent (the KPI row above); the over-the-window
    # stats below are all derived from the windowed daily returns. BTC daily via Yahoo is the only
    # crypto path; the BTC headline KPI above covers the live figure.
    adf = data.assets()
    syms = adf["symbol"].tolist() if not adf.empty else []
    long_df = data.all_assets_daily(rng_start)

    if not syms or long_df.empty:
        st.info("No asset data yet. Run `mmi ingest` or `make demo` to populate the markets tab.")
    else:
        # ---- 1. Cross-asset leaderboard — over-the-window return + annualised vol -------------
        board = charts.cross_asset_leaderboard(long_df)
        if not board.empty:
            st.caption("📊 Over the selected range · sorted by return")
            lb_cols = st.columns(min(len(board), 3))
            for i, row in enumerate(board.itertuples(index=False)):
                with lb_cols[i % len(lb_cols)]:
                    dot = charts.asset_class_color(row.asset_class)
                    ret_color = charts.leaderboard_return_color(row.period_return)
                    st.markdown(
                        f"<div style='line-height:1.35'>"
                        f"<span style='color:{dot};font-size:1.2em'>●</span> "
                        f"<b>{row.symbol}</b><br>"
                        f"<span style='color:{ret_color};font-size:1.15em;font-weight:600'>"
                        f"{row.period_return * 100:+.1f}%</span><br>"
                        f"<span style='color:{charts.PALETTE['muted']};font-size:0.85em'>"
                        f"vol {row.ann_vol * 100:.0f}%</span></div>",
                        unsafe_allow_html=True,
                    )
            st.divider()

        # ---- 2. Cross-asset performance, rebased to 0% at the window start --------------------
        perf = charts.rebased_performance(long_df)
        if not perf.empty:
            _chart(charts.rebased_performance_chart(perf, height=320))

        # ---- 3. Correlation heatmap (with the <30-obs guard) ---------------------------------
        corr = charts.correlation_matrix(long_df)
        if corr is None:
            st.caption(charts.CORR_TOO_SHORT)
        else:
            _chart(charts.correlation_heatmap(corr, height=320))
            takeaway = charts.correlation_takeaway(corr)
            if takeaway:
                st.caption(takeaway)

        # ---- 4. Per-asset drill-down — the original single-asset price + vol view ------------
        st.divider()
        st.caption("🔎 Per-asset detail")
        sym = st.selectbox("Asset", syms, index=syms.index("SPY") if "SPY" in syms else 0)
        # vol_20d / ma_50 are precomputed over FULL history in the mart — slice them for display
        # (so a short window still shows a correct MA at its left edge), never recomputed here.
        d = data.asset_daily(sym, rng_start)
        if not d.empty:
            mc1, mc2 = st.columns(2)
            with mc1:
                _chart(charts.price_chart(d, sym))
            with mc2:
                _chart(charts.vol_chart(d, sym))
        else:
            st.info(
                f"No daily price data for {sym} yet. Run `mmi ingest` (or `make demo`) to populate."
            )

with tab_macro:
    # Macro monitor: a headline snapshot (latest, range-independent), then a category selector that
    # swaps in a small-multiples grid of that theme's indicators (windowed by the range), then
    # always-on cross-series context (yield curve + recession risk). Reads the configured catalogue
    # for friendly labels + grouping (the mart only stores the raw FRED series_id).
    catalog = data.macro_catalog()
    present = set(data.macro_ids())
    cat = [c for c in catalog if c["id"] in present]
    by_id = {c["id"]: c for c in cat}
    mm_view = data.market_macro(rng_start)

    if not cat and mm_view.empty:
        st.info(
            "No macro series yet. Run `mmi ingest` (or `make demo`) to pull the FRED indicators. "
            "In the daily-cron partial state this tab populates once the first full ingest runs."
        )
    else:

        def _fmt_macro(v: float, u: str) -> str:
            if "%" in u:
                return f"{v:,.1f}%"
            if u == "pp":
                return f"{v:+,.2f}"
            if abs(v) >= 1000:
                return f"{v:,.0f}"
            return f"{v:,.1f}"

        def _fmt_macro_delta(chg: float, u: str) -> str:
            # A one-period change in a percent/rate series (UNRATE, yields, GDP growth, debt/GDP)
            # is in percentage POINTS, so suffix " pp"; index/$ deltas stay bare (the metric
            # label already names the series, and delta_color is off so it's context, not signal).
            if "%" in u or u == "pp":
                return f"{chg:+,.2f} pp"
            return f"{chg:+,.2f}"

        # ---- Snapshot strip: headline gauges, LATEST value (NOT filtered by the range). Deltas are
        # neutral (delta_color='off') — for macro, up/down isn't inherently good or bad. ----
        _MACRO_HEADLINE = [
            "A191RL1Q225SBEA",
            "UNRATE",
            "VIXCLS",
            "T10Y2Y",
            "FEDFUNDS",
            "GFDEGDQ188S",
        ]
        snap = [by_id[i] for i in _MACRO_HEADLINE if i in by_id]
        if snap:
            for col, c in zip(st.columns(len(snap)), snap, strict=True):
                s = data.macro(c["id"])  # full series → latest headline value, range-independent
                if s.empty:
                    continue
                chg = s["change"].dropna()
                with col:
                    st.metric(
                        c["label"],
                        _fmt_macro(float(s["value"].iloc[-1]), c["units"]),
                        delta=(
                            _fmt_macro_delta(float(chg.iloc[-1]), c["units"])
                            if not chg.empty
                            else None
                        ),
                        delta_color="off",
                    )
            st.divider()

        # ---- Category selector → small-multiples grid (each chart windowed by the range) ----
        _CAT_ORDER = [
            "Growth & activity",
            "Inflation",
            "Labor",
            "Rates & curve",
            "Fiscal",
            "Money & liquidity",
            "Risk & conditions",
            "Commodities & FX",
            "Other",
        ]
        cats_present = [k for k in _CAT_ORDER if any(c["category"] == k for c in cat)]
        if cats_present:
            sel_cat = (
                st.segmented_control(
                    "Category", cats_present, default=cats_present[0], key="macro_cat"
                )
                or cats_present[0]
            )
            gcols = st.columns(2)
            for i, c in enumerate(c for c in cat if c["category"] == sel_cat):
                with gcols[i % 2]:
                    df = data.macro(c["id"], rng_start)
                    if df.empty:
                        st.caption(f"{c['label']} — no data in this range")
                    else:
                        _chart(charts.macro_chart(df, c["label"], c["units"], height=240))
        macro_caption = data.macro_source_caption(is_sample)
        if macro_caption:
            st.caption(macro_caption)

        # ---- Always-on context: the yield-curve spread (cross-series composite) ----
        if not mm_view.empty:
            st.divider()
            st.caption("📌 Always-on context")
            _chart(charts.yield_curve_chart(mm_view))

    # ---- Recession-risk panel -----------------------------------------------
    # Macro CONTEXT only — not a return/price forecast (Contract E, §8).
    # The panel is always rendered (even when the main macro series are empty) because
    # fct_recession_risk is an independent mart built from the yield-curve data.
    rr = data.recession_risk(rng_start)
    with st.expander("📉 Recession-risk probability (yield-curve model)", expanded=not rr.empty):
        if rr.empty:
            st.info(
                "Recession-risk data not available yet. "
                "The `fct_recession_risk` mart is built during `mmi ingest` → `dbt build`. "
                "Run `make demo` or `mmi ingest` to populate."
            )
        else:
            _chart(charts.recession_risk_chart(rr))
        # Caveats are always shown so the panel reads as honest context even before data arrives.
        st.caption(charts._RECESSION_RISK_CAVEATS)
        rr_caption = charts.recession_risk_caption(is_sample)
        if rr_caption:
            st.caption(rr_caption)

with tab_ml:
    metrics = data.model_metrics()
    fc = data.ml_forecast()
    if metrics.empty:
        st.info(
            "No ML results yet. "
            "This is expected in the daily-cron partial state: the ML step runs only in the full "
            "local refresh (`make refresh-full`). Run `make ml` (or `mmi ml`) locally to train "
            "and commit updated forecasts."
        )
    else:
        # ---- Headline: HAR realized-volatility model (Contract E) ---------------
        # The certified ML target is next-week realized-volatility forecasting for SPY.
        # Verdict text is ALWAYS sourced from skill_verdict() (via charts.vol_skill_verdict_text)
        # — never re-derived here. The 'beats baseline OOS' language only appears when cleared.
        st.subheader("Volatility forecast — HAR model (headline)")
        st.caption(charts.ML_SCOPE_CAPTION)
        verdict_text = charts.vol_skill_verdict_text(metrics, symbol="SPY")
        if "no demonstrated out-of-sample edge" in verdict_text:
            st.warning(verdict_text)
        else:
            st.success(verdict_text)

        vol_col1, vol_col2 = st.columns([1, 1])
        with vol_col1:
            _chart(charts.vol_skill_r2_chart(metrics, symbol="SPY"))
        with vol_col2:
            _chart(charts.vol_skill_qlike_chart(metrics, symbol="SPY"))

        # Predicted next-week volatility and training date — read from the existing accessors.
        # The forecast lookup filters on BOTH model AND symbol (see vol_forecast_value) so a
        # future multi-symbol ML run can't surface another asset's row in the SPY headline.
        pred_vol = charts.vol_forecast_value(fc, symbol="SPY")
        rv_metrics = (
            metrics[(metrics["model"] == "rv_har") & (metrics["symbol"] == "SPY")]
            if not metrics.empty
            else metrics
        )
        fc_col1, fc_col2 = st.columns([1, 1])
        with fc_col1:
            if pred_vol is not None:
                st.metric(
                    "SPY predicted next-week realized vol (annualised)",
                    f"{pred_vol * 100:.2f}%",
                )
            else:
                st.caption("No SPY volatility forecast available yet.")
        with fc_col2:
            if not rv_metrics.empty and "trained_at" in rv_metrics.columns:
                trained_at = rv_metrics["trained_at"].dropna()
                if not trained_at.empty:
                    st.caption(f"Model trained {trained_at.iloc[0]}")

        # ---- Locked holdout (vol) — an honest extra OOS readout, NOT the gate ----------------
        # Absent on small-data (the holdout is skipped) and pre-re-run snapshots; render only when
        # the holdout_* rows are present. Never feeds skill_verdict() / the go-live gate.
        vol_holdout = charts.holdout_readout(metrics, model="rv_har", symbol="SPY")
        if vol_holdout is not None:
            st.caption(charts.HOLDOUT_CAPTION)
            hv1, hv2, hv3 = st.columns(3)
            if "holdout_oos_r2" in vol_holdout:
                hv1.metric("Holdout OOS R²", f"{vol_holdout['holdout_oos_r2']:.3f}")
            if "holdout_qlike_skill_ratio" in vol_holdout:
                hv2.metric(
                    "Holdout QLIKE skill ratio", f"{vol_holdout['holdout_qlike_skill_ratio']:.3f}"
                )
            if "holdout_n_obs" in vol_holdout:
                hv3.metric("Holdout obs", f"{vol_holdout['holdout_n_obs']:.0f}")

        reg_view = data.regimes("SPY", rng_start)
        if not reg_view.empty:
            _chart(charts.regime_chart(reg_view, "SPY"))

        st.divider()

        # ---- Secondary: next-day direction model (honestly demoted) ---------------
        # No demonstrated short-horizon edge — shown as an honest secondary, not the gate.
        st.subheader("Next-day direction model (honest secondary)")
        st.caption(
            "This model targets next-day SPY price direction. "
            "Short-horizon equity direction is near-noise; this model carries "
            "**no demonstrated out-of-sample edge** and is not used as a go-live gate."
        )
        _chart(charts.direction_skill_chart(metrics, symbol="SPY"))

        # ---- Locked holdout (direction) — honest extra OOS readout, NOT gated ----------------
        # Mirrors direction_skill_chart's "not the vol model" filter so a future direction-model
        # rename can't drop it. Absent on small-data / pre-re-run snapshots → render nothing.
        dir_holdout = charts.holdout_readout(metrics, symbol="SPY", exclude_model="rv_har")
        if dir_holdout is not None:
            st.caption(charts.HOLDOUT_CAPTION)
            hd1, hd2, hd3 = st.columns(3)
            if "holdout_dir_acc" in dir_holdout:
                hd1.metric("Holdout dir. accuracy", f"{dir_holdout['holdout_dir_acc'] * 100:.1f}%")
            if "holdout_baseline_dir_acc" in dir_holdout:
                hd2.metric(
                    "Holdout baseline accuracy",
                    f"{dir_holdout['holdout_baseline_dir_acc'] * 100:.1f}%",
                )
            if "holdout_n_obs" in dir_holdout:
                hd3.metric("Holdout obs", f"{dir_holdout['holdout_n_obs']:.0f}")

with tab_ai:
    brief = data.latest_brief()
    if brief.empty:
        st.info(
            "No AI brief yet. "
            "This is expected in the daily-cron partial state: the brief is generated only in "
            "the full local refresh. Run `make ai` (or `mmi ai`) locally — it works offline "
            "without an LLM key (falls back to a deterministic template)."
        )
    else:
        st.markdown(brief["brief"].iloc[0])
        st.caption(f"Generated by `{brief['engine'].iloc[0]}` · {brief['created_at'].iloc[0]}")
        st.caption(
            "Briefs refresh weekly (Mon 04:00 UTC, on the full pipeline run); the daily refresh "
            "preserves the latest brief rather than regenerating it."
        )
    # Regenerating needs a writable DB + an LLM key — neither exists in the public, read-only
    # snapshot deploy, so the control is hidden there.
    if not settings.snapshot_mode and st.button("🔄 Regenerate brief"):
        from mmi.ai.narrative import generate_brief

        con = connect()
        try:
            generate_brief(con)
        finally:
            con.close()
        st.cache_data.clear()
        st.rerun()

with tab_portfolio:
    # Phase D: pick the backtest window once, here, and thread it into every panel below — a single
    # choke point so every chart/table is for exactly one window (no cross-window aggregation).
    present_windows = data.portfolio_windows()
    if not present_windows:
        st.info(
            "No portfolio backtest yet. "
            "This is expected in the daily-cron partial state: the portfolio backtest runs only "
            "in the full local refresh (`make refresh-full` or `mmi portfolio`). "
            "The committed `data/public/` snapshot will include portfolio results once the "
            "owner's next full run completes."
        )
    else:
        window_id = present_windows[0]
        if len(present_windows) > 1:
            window_id = st.radio(
                "Backtest window",
                present_windows,
                format_func=lambda w: _WINDOW_LABELS.get(w, w),
                horizontal=True,
                key="portfolio_window",
            )
            st.caption(
                "⚠️ inc-BTC vs ex-BTC@2002 differs in BOTH universe AND period — use the BTC-impact "
                "section below (the ex/inc 2015 pair) for the clean, same-period BTC comparison. "
                "Volatility regimes are cut within each window, so regime labels aren't comparable "
                "across windows."
            )
        pf = data.portfolio_returns(window_id, rng_start)
        st.caption(
            "Walk-forward backtest: three allocation strategies vs a 60/40 benchmark — same dates, "
            "monthly rebalancing and round-trip costs, so the comparison is like-for-like."
        )
        # Findings, promoted to the top: the honest bootstrap verdict before any chart.
        pairs = data.portfolio_strategy_pairs(window_id)
        if not pairs.empty:
            st.info("📊 " + charts.distinguishability_verdict(pairs))
        _chart(charts.portfolio_cumulative_chart(pf))
        if rng_start:
            st.caption("Cumulative return is rebased to 0% at the start of the selected range.")

        # ---- secondary sections (collapsible) --------------------------------
        with st.expander("📉 Drawdown & rolling Sharpe", expanded=False):
            cda, cdb = st.columns(2)
            with cda:
                _chart(charts.portfolio_drawdown_chart(pf))
            with cdb:
                _chart(charts.portfolio_sharpe_chart(pf))
            st.dataframe(
                charts.portfolio_summary(pf).style.format(
                    {
                        "Total return": "{:+.1%}",
                        "Max drawdown": "{:.1%}",
                        "Ann. vol": "{:.1%}",
                        "Sharpe (252d)": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

        # Risk-adjusted scorecard: full-sample Sharpe + bootstrap CIs + pairwise distinguishability.
        stats = data.portfolio_strategy_stats(window_id)
        if not stats.empty:
            ci = int(round(stats["ci_pct"].iloc[0] * 100))
            with st.expander(
                f"📊 Risk-adjusted scorecard — Sharpe with {ci}% bootstrap CI", expanded=False
            ):
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.dataframe(
                        charts.portfolio_scorecard(stats).style.format("{:.2f}"),
                        use_container_width=True,
                    )
                with sc2:
                    if not pairs.empty:
                        st.dataframe(
                            charts.portfolio_pairs_table(pairs).style.format(
                                {"Δ Sharpe": "{:.2f}", "CI low": "{:.2f}", "CI high": "{:.2f}"}
                            ),
                            use_container_width=True,
                        )
                st.caption(
                    f"Stationary block-bootstrap ({stats['n_boot'].iloc[0]:,} resamples, "
                    f"{stats['n_obs'].iloc[0]} obs). Distinguishable = Sharpe-diff CI excludes 0."
                )

        # Return attribution — what drove each strategy's return.
        attr = data.portfolio_attribution(window_id)
        if not attr.empty:
            with st.expander("📈 Return attribution", expanded=False):
                astrat = st.selectbox(
                    "Strategy", sorted(attr["strategy"].unique()), key="attribution_strategy"
                )
                _chart(charts.attribution_chart(attr, astrat))

        # Regime-conditional performance — behaviour across market volatility regimes.
        regime = data.portfolio_regime_performance(window_id)
        if not regime.empty:
            with st.expander("🌡️ Performance by market volatility regime", expanded=False):
                _chart(charts.regime_sharpe_chart(regime))
                st.caption(
                    "Market regime = SPY 20-day-vol terciles; "
                    "stats over each strategy's invested days."
                )

        # ML experiment: did the forecast add value? (the gate makes mvo_ml ≈ mvo_histmean legible)
        gate = data.portfolio_ml_gate(window_id, rng_start)
        if not gate.empty:
            with st.expander("🔬 ML experiment — does the forecast add value?", expanded=False):
                st.info("🔬 " + charts.ml_verdict(gate, pairs))
                _chart(charts.ml_gate_chart(gate))
                st.caption(
                    "forecast_weight is the share mvo_ml puts on the ML forecast over the "
                    "historical-mean prior, gated point-in-time by the forecast's realised skill. "
                    "Pre-registered: expected to stay low."
                )

        # BTC impact: the same-period (2015) paired comparison — independent of the window picker.
        btc_effect = data.portfolio_btc_effect()
        if not btc_effect.empty:
            with st.expander(
                "🪙 BTC impact — does adding BTC change risk-adjusted return?", expanded=False
            ):
                st.info("🪙 " + charts.btc_effect_verdict(btc_effect))
                _chart(charts.btc_effect_chart(btc_effect))
                st.caption(
                    "Sharpe(inc-BTC@2015) − Sharpe(ex-BTC@2015): same dates ± BTC, with a "
                    "paired block-bootstrap 90% CI. The 60/40 benchmark (never holds BTC) is "
                    "exactly zero — a check that the comparison is genuinely paired. "
                    "BTC's weekend moves fold into the next trading day, "
                    "so its standalone daily vol is understated here."
                )

# --------------------------------------------------------------------------- footer
st.divider()
_footer_col1, _footer_col2, _footer_col3 = st.columns([2, 2, 2])
with _footer_col1:
    st.caption(
        "Source: [github.com/mathsisbest/macro-scope](https://github.com/mathsisbest/macro-scope)"
    )
with _footer_col2:
    st.caption("Built by **mathsisbest** · not investment advice")
with _footer_col3:
    if as_of:
        st.caption(f"Data as of {as_of}")
    else:
        st.caption("No data loaded")
