import streamlit as st
import pandas as pd
from dashboard import data
from dashboard.components import charts
from dashboard.theme import PLOTLY_CONFIG

def _chart(fig, **kwargs):
    kwargs.setdefault("config", PLOTLY_CONFIG)
    st.plotly_chart(fig, width="stretch", **kwargs)

def render_macro_tab(rng_start, as_of):
    catalog = data.macro_catalog()
    present = set(data.macro_ids())
    cat = [c for c in catalog if c["id"] in present]
    by_id = {c["id"]: c for c in cat}
    mm_view = data.market_macro(rng_start)
    is_sample = data.is_sample_data()

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
            if "%" in u or u == "pp":
                return f"{chg:+,.2f} pp"
            return f"{chg:+,.2f}"

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
            for chunk_start in range(0, len(snap), 3):
                chunk = snap[chunk_start : chunk_start + 3]
                for col, c in zip(st.columns(len(chunk)), chunk, strict=True):
                    s = data.macro(c["id"])
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
            st.markdown("</div>", unsafe_allow_html=True)
            st.divider()

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
            sel_cat = st.selectbox("Category", cats_present, key="macro_cat")
            gcols = st.columns(2)
            for i, c in enumerate(c for c in cat if c["category"] == sel_cat):
                with gcols[i % 2]:
                    df = data.macro(c["id"], rng_start)
                    if df.empty:
                        st.caption(f"{c['label']} — no data in this range")
                    else:
                        _chart(charts.macro_chart(df, c["label"], c["units"], height=200))
                        if c["id"] == "VIXCLS":
                            with st.expander("ℹ️ About VIX"):
                                st.caption("The VIX measures 30-day expected S&P 500 volatility implied by options. Spikes above 30 historically coincide with market corrections.")
                        elif c["id"] == "NFCI":
                            with st.expander("ℹ️ About NFCI"):
                                st.caption("The National Financial Conditions Index measures stress in money, debt, and equity markets. Positive values = tighter than average conditions.")

        macro_caption = data.macro_source_caption(is_sample)
        if macro_caption:
            st.caption(macro_caption)

        if not mm_view.empty:
            st.divider()
            st.caption("📌 Always-on context")
            _chart(charts.yield_curve_chart(mm_view))
            with st.expander("ℹ️ About Yield Curve"):
                st.caption("An inverted yield curve (negative spread) has preceded every US recession since 1960, typically by 12–18 months.")

    # Valuation Section (Task 1.8)
    st.divider()
    st.subheader("📊 Valuation")
    val_df = data.valuation_data(rng_start)
    if not val_df.empty and "erp" in val_df.columns:
        st.caption("ERP (Equity Risk Premium) = S&P 500 earnings yield (1/CAPE) minus 10Y Treasury yield. It measures the excess return investors demand for holding equities over risk-free bonds.")
        fig = charts.go.Figure()
        fig.add_scatter(x=val_df["date"], y=val_df["erp"], name="ERP (%)", line=dict(color=charts.PALETTE["accent"]))
        fig.update_layout(title=dict(text="Equity Risk Premium (ERP)", font=charts._TITLE_FONT))
        charts._apply_axis_fonts(fig)
        _chart(charts.style_fig(fig, height=charts.HEIGHT_DEFAULT))
    else:
        st.info("Valuation data not available. (CAPE data missing).")

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
            with st.expander("ℹ️ About Recession Risk"):
                st.caption("Composite probability model based on yield curve inversion, unemployment trend, and financial conditions.")
        st.caption(charts._RECESSION_RISK_CAVEATS)
        rr_caption = charts.recession_risk_caption(is_sample)
        if rr_caption:
            st.caption(rr_caption)
