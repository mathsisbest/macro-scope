"""Macro Tab Module."""

from __future__ import annotations

import streamlit as st
from dashboard import data
from dashboard.components import charts, glossary


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


def render_macro_tab(rng_start: str | None, is_sample: bool | None, chart_wrapper) -> None:
    """Render the Macro tab."""
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
                        chart_wrapper(charts.macro_chart(df, c["label"], c["units"], height=200))
        macro_caption = data.macro_source_caption(is_sample)
        if macro_caption:
            st.caption(macro_caption)

        if not mm_view.empty:
            st.divider()
            st.caption("📌 Always-on context")
            chart_wrapper(charts.yield_curve_chart(mm_view))

    rr = data.recession_risk(rng_start)
    with st.expander("📉 Recession-risk probability (yield-curve model)", expanded=not rr.empty):
        if rr.empty:
            st.info(
                "Recession-risk data not available yet. "
                "The `fct_recession_risk` mart is built during `mmi ingest` → `dbt build`. "
                "Run `make demo` or `mmi ingest` to populate."
            )
        else:
            chart_wrapper(charts.recession_risk_chart(rr))
        st.caption(charts._RECESSION_RISK_CAVEATS)
        rr_caption = charts.recession_risk_caption(is_sample)
        if rr_caption:
            st.caption(rr_caption)
        glossary.glossary_tooltip("yield_curve_spread")
