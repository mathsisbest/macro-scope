import streamlit as st
import pandas as pd
from dashboard import data
from dashboard.components import charts
from dashboard.theme import PLOTLY_CONFIG

def _chart(fig, **kwargs):
    """Thin wrapper so every chart gets the mobile-safe config (no scroll-zoom, no modebar)."""
    kwargs.setdefault("config", PLOTLY_CONFIG)
    st.plotly_chart(fig, width="stretch", **kwargs)

def render_markets_tab(rng_start, as_of):
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
            st.markdown('<div class="leaderboard">', unsafe_allow_html=True)
            lb_cols = st.columns(min(len(board), 3))
            for i, row in enumerate(board.itertuples(index=False)):
                with lb_cols[i % len(lb_cols)]:
                    if row.asset_class:
                        dot = charts.asset_class_color(row.asset_class)
                    else:
                        dot = charts.PALETTE["series"][i % 6]
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
