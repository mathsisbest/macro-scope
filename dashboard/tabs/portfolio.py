"""Portfolio Tab Module."""

from __future__ import annotations

import streamlit as st
from dashboard import data
from dashboard.components import charts

_WINDOW_LABELS = {
    "ex_btc_2002": "~2004–present · ex-BTC",
    "ex_btc_2015": "2015–present · ex-BTC (BTC era)",
    "inc_btc_2015": "2015–present · incl. BTC",
}


def render_portfolio_tab(rng_start: str | None, chart_wrapper) -> None:
    """Render the Portfolio tab."""
    present_windows = data.portfolio_windows()
    if not present_windows:
        st.info(
            "No portfolio backtest yet. "
            "This is expected in the daily-cron partial state: the portfolio backtest runs only "
            "in the full local refresh (`make refresh-full` or `mmi portfolio`). "
            "The committed `data/public/` snapshot will include portfolio results once the "
            "owner's next full run completes."
        )
        return

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
    pairs = data.portfolio_strategy_pairs(window_id)
    if not pairs.empty:
        st.info("📊 " + charts.distinguishability_verdict(pairs))
    chart_wrapper(charts.portfolio_cumulative_chart(pf, height=320))
    if rng_start:
        st.caption("Cumulative return is rebased to 0% at the start of the selected range.")

    with st.expander("📉 Drawdown & rolling Sharpe", expanded=False):
        cda, cdb = st.columns(2)
        with cda:
            chart_wrapper(charts.portfolio_drawdown_chart(pf, height=260))
        with cdb:
            chart_wrapper(charts.portfolio_sharpe_chart(pf, height=260))
        st.dataframe(
            charts.portfolio_summary(pf).style.format(
                {
                    "Total return": "{:+.1%}",
                    "Max drawdown": "{:.1%}",
                    "Ann. vol": "{:.1%}",
                    "Sharpe (252d)": "{:.2f}",
                }
            ),
            width="stretch",
        )

    stats = data.portfolio_strategy_stats(window_id)
    if not stats.empty:
        ci = int(round(stats["ci_pct"].iloc[0] * 100))
        with st.expander(
            f"📊 Risk-adjusted scorecard — Sharpe with {ci}% bootstrap CI", expanded=True
        ):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.dataframe(
                    charts.portfolio_scorecard(stats).style.format("{:.2f}"),
                    width="stretch",
                )
            with sc2:
                if not pairs.empty:
                    st.dataframe(
                        charts.portfolio_pairs_table(pairs).style.format(
                            {"Δ Sharpe": "{:.2f}", "CI low": "{:.2f}", "CI high": "{:.2f}"}
                        ),
                        width="stretch",
                    )
            st.caption(
                f"Stationary block-bootstrap ({stats['n_boot'].iloc[0]:,} resamples, "
                f"{stats['n_obs'].iloc[0]} obs). Distinguishable = Sharpe-diff CI excludes 0."
            )

    attr = data.portfolio_attribution(window_id)
    if not attr.empty:
        with st.expander("📈 Return attribution", expanded=False):
            astrat = st.selectbox(
                "Strategy", sorted(attr["strategy"].unique()), key="attribution_strategy"
            )
            chart_wrapper(charts.attribution_chart(attr, astrat))

    regime = data.portfolio_regime_performance(window_id)
    if not regime.empty:
        with st.expander("🌡️ Performance by market volatility regime", expanded=False):
            chart_wrapper(charts.regime_sharpe_chart(regime))
            st.caption(
                "Market regime = SPY 20-day-vol terciles; stats over each strategy's invested days."
            )

    gate = data.portfolio_ml_gate(window_id, rng_start)
    if not gate.empty:
        with st.expander("🔬 ML experiment — does the forecast add value?", expanded=False):
            st.info("🔬 " + charts.ml_verdict(gate, pairs))
            chart_wrapper(charts.ml_gate_chart(gate))
            st.caption(
                "forecast_weight is the share mvo_ml puts on the ML forecast over the "
                "historical-mean prior, gated point-in-time by the forecast's realised skill. "
                "Pre-registered: expected to stay low."
            )

    btc_effect = data.portfolio_btc_effect()
    if not btc_effect.empty:
        with st.expander(
            "🪙 BTC impact — does adding BTC change risk-adjusted return?", expanded=False
        ):
            st.info("🪙 " + charts.btc_effect_verdict(btc_effect))
            chart_wrapper(charts.btc_effect_chart(btc_effect))
            st.caption(
                "Sharpe(inc-BTC@2015) − Sharpe(ex-BTC@2015): same dates ± BTC, with a "
                "paired block-bootstrap 90% CI. The 60/40 benchmark (never holds BTC) is "
                "exactly zero — a check that the comparison is genuinely paired. "
                "BTC's weekend moves fold into the next trading day, "
                "so its standalone daily vol is understated here."
            )
