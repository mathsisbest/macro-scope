"""Markets & Macro Intelligence — Streamlit dashboard (BI layer).

Run: `make dashboard` / `make demo`, or `streamlit run dashboard/app.py` directly — this file
puts the repo root on sys.path so `from dashboard import ...` resolves everywhere (local and
Streamlit Community Cloud, which otherwise only has this file's own dir on the path).
Reads the dbt marts from DuckDB; everything visual is defined in code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud runs this file with only its own directory on sys.path (not the
# repo root), so the repo-root `dashboard` package isn't importable. Put the repo root first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import data  # noqa: E402
from dashboard.components import charts  # noqa: E402
from dashboard.components.kpi import metric_row  # noqa: E402
from dashboard.theme import inject_css  # noqa: E402

from mmi.settings import settings  # noqa: E402
from mmi.utils.db import connect  # noqa: E402

# --------------------------------------------------------------------------- page config
_FAVICON = Path(__file__).resolve().parent / "assets" / "favicon.png"
st.set_page_config(
    page_title="Markets & Macro Intelligence",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "📈",
    layout="wide",
)
inject_css()

# --------------------------------------------------------------------------- hero / header
st.title("📈 Markets & Macro Intelligence")
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
FRED, CoinGecko, World Bank, DuckDB, scikit-learn, a local or serverless LLM).

**Data sources**

- **Yahoo Finance (unofficial)** — equities, ETFs, FX daily OHLCV.
  Unofficial API; not endorsed by Yahoo Finance.
- **FRED — Federal Reserve Bank of St. Louis** — macro series (CPI, unemployment,
  Fed Funds rate, yield curve). [fred.stlouisfed.org](https://fred.stlouisfed.org/)
- **CoinGecko** — crypto intraday prices. Free-tier API; rate-limited.
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

This dashboard is a data-engineering and machine-learning project.
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
    st.subheader("⚙️ Pipeline health")
    runs = data.pipeline_runs()
    if not runs.empty:
        st.dataframe(runs, hide_index=True, use_container_width=True)
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
kpis: list[dict] = []
csyms = data.crypto_symbols()
if csyms:
    cdf = data.crypto_intraday(csyms[0])
    if len(cdf) > 25:
        last, prev = cdf["price_usd"].iloc[-1], cdf["price_usd"].iloc[-25]
        kpis.append(
            {
                "label": f"{csyms[0].title()} (USD)",
                "value": f"${last:,.0f}",
                "delta": f"{(last / prev - 1) * 100:+.1f}% 24h",
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
if not mm.empty and mm["yield_curve_10y_2y"].notna().any():
    spread = mm["yield_curve_10y_2y"].dropna().iloc[-1]
    kpis.append({"label": "10Y−2Y spread", "value": f"{spread:+.2f} pp"})

if kpis:
    metric_row(kpis)

st.divider()

# --------------------------------------------------------------------------- tabs
# Human labels for the Phase-D backtest windows (the selector in the Portfolio tab).
_WINDOW_LABELS = {
    "ex_btc_2002": "2002–present · ex-BTC",
    "ex_btc_2015": "2015–present · ex-BTC (BTC era)",
    "inc_btc_2015": "2015–present · incl. BTC",
}

tab_mkt, tab_macro, tab_ml, tab_ai, tab_portfolio = st.tabs(
    ["Markets", "Macro", "ML forecast", "AI brief", "Portfolio"]
)

with tab_mkt:
    adf = data.assets()
    non_crypto = adf[adf["asset_class"] != "crypto"]["symbol"].tolist() if not adf.empty else []
    col1, col2 = st.columns(2)
    with col1:
        if non_crypto:
            sym = st.selectbox(
                "Asset", non_crypto, index=non_crypto.index("SPY") if "SPY" in non_crypto else 0
            )
            d = data.asset_daily(sym)
            if not d.empty:
                st.plotly_chart(charts.price_chart(d, sym), use_container_width=True)
                st.plotly_chart(charts.vol_chart(d, sym), use_container_width=True)
            else:
                st.info(
                    f"No daily price data for {sym} yet. "
                    "Run `mmi ingest` (or `make demo`) to populate."
                )
        else:
            st.info(
                "No asset data yet. Run `mmi ingest` or `make demo` to populate the markets tab."
            )
    with col2:
        if csyms:
            c = st.selectbox("Crypto", csyms)
            cd = data.crypto_intraday(c)
            if not cd.empty:
                st.plotly_chart(charts.crypto_chart(cd, c), use_container_width=True)
            else:
                st.info(
                    f"No intraday data for {c} yet. Run `mmi ingest` (or `make demo`) to populate."
                )
        else:
            st.info(
                "No crypto data yet. "
                "This is normal during the daily-cron partial state — "
                "crypto data arrives on the next full ingest."
            )

with tab_macro:
    ids = data.macro_ids()
    if ids:
        mid = st.selectbox("Series", ids)
        md = data.macro(mid)
        if not md.empty:
            st.plotly_chart(charts.macro_chart(md, mid), use_container_width=True)
        else:
            st.info(f"No data for series {mid} yet.")
    else:
        st.info(
            "No macro series yet. "
            "Run `mmi ingest` (or `make demo`) to pull FRED / World Bank indicators. "
            "In the daily-cron partial state this tab populates once the first full ingest runs."
        )
    if not mm.empty:
        st.plotly_chart(charts.yield_curve_chart(mm), use_container_width=True)
    # Source line for the macro charts. Live FRED data (every series here — CPIAUCSL, UNRATE, DGS10,
    # DGS2, FEDFUNDS — and the 10Y−2Y curve are FRED) earns the FRED attribution; sample data must
    # NOT be attributed to FRED (it's synthetic). Decision lives in a pure, tested helper.
    macro_caption = data.macro_source_caption(is_sample)
    if (ids or not mm.empty) and macro_caption:
        st.caption(macro_caption)

    # ---- Recession-risk panel -----------------------------------------------
    # Macro CONTEXT only — not a return/price forecast (Contract E, §8).
    # The panel is always rendered (even when the main macro series are empty) because
    # fct_recession_risk is an independent mart built from the yield-curve data.
    rr = data.recession_risk()
    with st.expander("📉 Recession-risk probability (yield-curve model)", expanded=not rr.empty):
        if rr.empty:
            st.info(
                "Recession-risk data not available yet. "
                "The `fct_recession_risk` mart is built during `mmi ingest` → `dbt build`. "
                "Run `make demo` or `mmi ingest` to populate."
            )
        else:
            st.plotly_chart(charts.recession_risk_chart(rr), use_container_width=True)
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
            st.plotly_chart(
                charts.vol_skill_r2_chart(metrics, symbol="SPY"), use_container_width=True
            )
        with vol_col2:
            st.plotly_chart(
                charts.vol_skill_qlike_chart(metrics, symbol="SPY"), use_container_width=True
            )

        # Predicted next-week volatility and training date — read from the existing accessors.
        rv_fc = fc[fc["model"] == "rv_har"] if not fc.empty and "model" in fc.columns else fc
        rv_metrics = (
            metrics[(metrics["model"] == "rv_har") & (metrics["symbol"] == "SPY")]
            if not metrics.empty
            else metrics
        )
        fc_col1, fc_col2 = st.columns([1, 1])
        with fc_col1:
            if not rv_fc.empty and "predicted_next_return" in rv_fc.columns:
                pred_vol = rv_fc["predicted_next_return"].iloc[0]
                st.metric(
                    "SPY predicted next-week realized vol (annualised)",
                    f"{pred_vol * 100:.2f}%",
                )
        with fc_col2:
            if not rv_metrics.empty and "trained_at" in rv_metrics.columns:
                trained_at = rv_metrics["trained_at"].dropna()
                if not trained_at.empty:
                    st.caption(f"Model trained {trained_at.iloc[0]}")

        if not reg.empty:
            st.plotly_chart(charts.regime_chart(reg, "SPY"), use_container_width=True)

        st.divider()

        # ---- Secondary: next-day direction model (honestly demoted) ---------------
        # No demonstrated short-horizon edge — shown as an honest secondary, not the gate.
        st.subheader("Next-day direction model (honest secondary)")
        st.caption(
            "This model targets next-day SPY price direction. "
            "Short-horizon equity direction is near-noise; this model carries "
            "**no demonstrated out-of-sample edge** and is not used as a go-live gate."
        )
        st.plotly_chart(
            charts.direction_skill_chart(metrics, symbol="SPY"), use_container_width=True
        )

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
        pf = data.portfolio_returns(window_id)
        st.caption(
            "Walk-forward backtest: three allocation strategies vs a 60/40 benchmark — same dates, "
            "monthly rebalancing and round-trip costs, so the comparison is like-for-like."
        )
        # Findings, promoted to the top: the honest bootstrap verdict before any chart.
        pairs = data.portfolio_strategy_pairs(window_id)
        if not pairs.empty:
            st.info("📊 " + charts.distinguishability_verdict(pairs))
        st.plotly_chart(charts.portfolio_cumulative_chart(pf), use_container_width=True)

        # ---- secondary sections (collapsible) --------------------------------
        with st.expander("📉 Drawdown & rolling Sharpe", expanded=False):
            cda, cdb = st.columns(2)
            with cda:
                st.plotly_chart(charts.portfolio_drawdown_chart(pf), use_container_width=True)
            with cdb:
                st.plotly_chart(charts.portfolio_sharpe_chart(pf), use_container_width=True)
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
                st.plotly_chart(charts.attribution_chart(attr, astrat), use_container_width=True)

        # Regime-conditional performance — behaviour across market volatility regimes.
        regime = data.portfolio_regime_performance(window_id)
        if not regime.empty:
            with st.expander("🌡️ Performance by market volatility regime", expanded=False):
                st.plotly_chart(charts.regime_sharpe_chart(regime), use_container_width=True)
                st.caption(
                    "Market regime = SPY 20-day-vol terciles; "
                    "stats over each strategy's invested days."
                )

        # ML experiment: did the forecast add value? (the gate makes mvo_ml ≈ mvo_histmean legible)
        gate = data.portfolio_ml_gate(window_id)
        if not gate.empty:
            with st.expander("🔬 ML experiment — does the forecast add value?", expanded=False):
                st.info("🔬 " + charts.ml_verdict(gate, pairs))
                st.plotly_chart(charts.ml_gate_chart(gate), use_container_width=True)
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
                st.plotly_chart(charts.btc_effect_chart(btc_effect), use_container_width=True)
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
        "Source: [github.com/mathsisbest/markets-macro-intelligence]"
        "(https://github.com/mathsisbest/markets-macro-intelligence)"
    )
with _footer_col2:
    st.caption("Built by **mathsisbest** · not investment advice")
with _footer_col3:
    if as_of:
        st.caption(f"Data as of {as_of}")
    else:
        st.caption("No data loaded")
