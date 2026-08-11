"""ML Forecast Tab Module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from dashboard import data
from dashboard.components import charts, glossary
from dashboard.tabs.ml_scenario import render_ml_scenario_tab


def render_ml_tab(chart_wrapper) -> None:
    """Render the ML Forecast tab."""
    metrics = data.model_metrics()
    fc = data.ml_forecast()
    if metrics.empty:
        st.info(
            "No ML results yet. "
            "This is expected in the daily-cron partial state: the ML step runs only in the full "
            "local refresh (`make refresh-full`). Run `make ml` (or `mmi ml`) locally to train "
            "and commit updated forecasts."
        )
        return

    st.subheader("ML Return Forecast — 20-Day Horizon Engine")
    st.caption(
        "Autotuned Gradient Boosting, LightGBM, and Regularized models trained on 75+ macro, "
        "volatility, and cross-asset ratio spread features (vol_rich_plus) over standardized "
        "20-day (1-month) forward return horizons."
    )
    st.markdown(
        glossary.tooltip_markdown("vol_rich_plus"),
        unsafe_allow_html=True,
    )
    st.info(
        "ℹ️ All forecasts target a standardized 20-day (1-month) horizon, "
        "calibrated via Bayesian shrinkage toward historical return means. "
        "Out-of-Sample evaluation metrics (IC, R², hit rate) "
        "reflect strict walk-forward performance."
    )

    CORE_SYMBOLS = ["SPY", "QQQ", "GLD", "TLT", "BTC"]
    return_fc = charts.return_forecast_table(fc)
    core_fc = (
        return_fc[return_fc["symbol"].isin(CORE_SYMBOLS)] if not return_fc.empty else return_fc
    )

    if not core_fc.empty:
        st.caption("Sorted by forecast return; each card uses the latest available row per asset.")
        for chunk_start in range(0, len(core_fc), 3):
            chunk = core_fc.iloc[chunk_start : chunk_start + 3]
            cols = st.columns(len(chunk))
            for col, row in zip(cols, chunk.itertuples(index=False), strict=True):
                pred = float(row.predicted_return)
                daily_mu = float(row.daily_mu) if pd.notna(row.daily_mu) else None
                daily_label = f"{daily_mu * 100:+.3f}%/day" if daily_mu is not None else "n/a"
                direction = "↑" if pred > 0 else "↓" if pred < 0 else "→"
                color = charts.leaderboard_return_color(pred)
                with col:
                    st.markdown(
                        f"""
                        <div class="forecast-card">
                          <div class="forecast-card__top">
                            <span class="forecast-card__symbol">{row.symbol}</span>
                            <span class="forecast-card__horizon">{int(row.horizon)}d</span>
                          </div>
                          <div class="forecast-card__value" style="color:{color}">
                            {direction} {pred * 100:+.2f}%
                          </div>
                          <div class="forecast-card__meta">
                            {daily_label} · as of {pd.to_datetime(row.as_of).date()}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with st.expander("Full universe forecast table (all assets)", expanded=False):
            st.dataframe(
                return_fc.assign(
                    predicted_return=lambda d: d["predicted_return"].map(lambda v: f"{v:.2%}"),
                    daily_mu=lambda d: d["daily_mu"].map(lambda v: f"{v:.3%}"),
                    horizon=lambda d: d["horizon"].astype("Int64"),
                ),
                hide_index=True,
                width="stretch",
            )

        render_ml_scenario_tab(chart_wrapper)

        st.divider()
        st.subheader("🎛️ Forecast confidence fan")
        st.caption(
            "Expected drift path (dashed-anchored at today = zero uncertainty) widening to ±1σ "
            "of the model's walk-forward out-of-sample residual error at the forecast horizon. "
            "A wider band means less demonstrated skill (lower OOS R²); R² is read from the "
            "latest `return_gb` evaluation for the asset."
        )
        fan_sym = st.selectbox("Fan chart asset", CORE_SYMBOLS, index=0)
        fan_row = return_fc[return_fc["symbol"] == fan_sym].head(1)
        if fan_row.empty:
            st.info("No forecast row for this asset yet.")
        else:
            row = fan_row.iloc[0]
            asset = data.asset_daily(fan_sym)
            sigma_daily = (
                float(asset["daily_return"].tail(252).std())
                if not asset.empty and "daily_return" in asset
                else np.nan
            )
            raw_fc = fc[fc["symbol"] == fan_sym]
            oos_r2 = (
                float(raw_fc["r2"].iloc[0])
                if not raw_fc.empty and "r2" in raw_fc and pd.notna(raw_fc["r2"].iloc[0])
                else None
            )
            if not np.isfinite(sigma_daily):
                st.info("No price history available to size the confidence band.")
            else:
                fan = charts.forecast_fan_points(
                    daily_mu=float(row["daily_mu"]),
                    horizon=float(row["horizon"]),
                    sigma_daily=sigma_daily,
                    oos_r2=oos_r2,
                )
                chart_wrapper(
                    charts.forecast_fan_chart(fan, fan_sym, as_of=row["as_of"], z=1.0, height=300)
                )

        st.divider()
        st.subheader("🎯 Active Market & Model Signals")
        st.caption(
            "Walk-forward out-of-sample evaluated models. Green = Deployed tilt (OOS R² > 0); "
            "Yellow = Directional / Regime Only; Red = Gated Out."
        )
        st.markdown(
            " · ".join(
                [
                    glossary.tooltip_markdown("oos_r2"),
                    glossary.tooltip_markdown("skill_gate"),
                ]
            ),
            unsafe_allow_html=True,
        )
        fc_table = core_fc
        metrics_data = data.model_metrics()
        if not fc_table.empty:
            tilts = []
            for row in fc_table.itertuples(index=False):
                sym = row.symbol
                pred = float(row.predicted_return)
                h = int(row.horizon)
                m_rows = (
                    metrics_data[
                        (metrics_data["model"] == "return_gb") & (metrics_data["symbol"] == sym)
                    ]
                    if not metrics_data.empty
                    else pd.DataFrame()
                )
                r2_val = (
                    m_rows[m_rows["metric"] == "r2"]["value"].iloc[0]
                    if not m_rows.empty and "r2" in m_rows["metric"].to_numpy()
                    else np.nan
                )
                dir_acc = (
                    m_rows[m_rows["metric"] == "direction_accuracy"]["value"].iloc[0]
                    if not m_rows.empty and "direction_accuracy" in m_rows["metric"].to_numpy()
                    else np.nan
                )

                if pd.notna(r2_val) and r2_val > 0:
                    status = "🟢 Active Tilt (Skill Cleared)"
                elif pd.notna(dir_acc) and dir_acc > 0.50:
                    status = "🟡 Directional / Regime Only"
                else:
                    status = "🔴 Gated Out (Noise)"

                tilts.append(
                    {
                        "Asset": sym,
                        "Horizon": f"{h}d",
                        "Forecast Return": f"{pred:+.2%}",
                        "OOS R²": f"{r2_val:+.3f}" if pd.notna(r2_val) else "N/A",
                        "Direction Accuracy": f"{dir_acc:.1%}" if pd.notna(dir_acc) else "N/A",
                        "Status": status,
                    }
                )
            st.dataframe(pd.DataFrame(tilts), hide_index=True, width="stretch")

        st.divider()
        st.subheader("⚡ Current Volatility Regimes")
        glossary.glossary_tooltip("vol_regime")
        reg_cols = st.columns(3)
        for idx, reg_sym in enumerate(["SPY", "TLT", "BTC"]):
            rv = data.regimes(reg_sym)
            if not rv.empty:
                cur_reg = str(rv["regime"].iloc[-1])
                with reg_cols[idx % 3]:
                    st.metric(label=f"{reg_sym} Regime", value=cur_reg)

        perf = charts.return_performance_table(metrics)
        if not perf.empty:
            st.divider()
            st.subheader("Return model performance")
            st.caption(
                "Per-asset diagnostics from `marts.model_metrics`: IC, R², direction "
                "accuracy, Sharpe, and observation count."
            )
            st.markdown(
                " · ".join(
                    [
                        glossary.tooltip_markdown("ic"),
                        glossary.tooltip_markdown("direction_accuracy"),
                        glossary.tooltip_markdown("sharpe"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            chart_wrapper(charts.return_performance_chart(perf, height=320))
            perf_fmt = perf.copy()
            for col_name, fmt_func in [
                ("ic", lambda v: "" if pd.isna(v) else f"{v:.3f}"),
                ("direction_accuracy", lambda v: "" if pd.isna(v) else f"{v:.1%}"),
                ("r2", lambda v: "" if pd.isna(v) else f"{v:.3f}"),
                ("sharpe", lambda v: "" if pd.isna(v) else f"{v:.2f}"),
                ("n_obs", lambda v: "" if pd.isna(v) else f"{v:,.0f}"),
            ]:
                if col_name in perf_fmt.columns:
                    perf_fmt[col_name] = perf_fmt[col_name].map(fmt_func)

            st.dataframe(
                perf_fmt.rename(
                    columns={
                        "symbol": "Asset",
                        "ic": "IC",
                        "direction_accuracy": "Direction accuracy",
                        "r2": "R²",
                        "sharpe": "Sharpe",
                        "n_obs": "Obs",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        st.divider()
        st.subheader("Regime breakdown")
        regime_perf = charts.return_regime_breakdown_table(metrics)
        if regime_perf.empty:
            st.info(
                "Regime-specific return metrics are not present in the current public "
                "snapshot. The app can render them once the ML pipeline persists "
                "`direction_accuracy_<regime>` rows by asset."
            )
        else:
            st.dataframe(
                regime_perf.assign(
                    direction_accuracy=lambda d: d["direction_accuracy"].map(
                        lambda v: "" if pd.isna(v) else f"{v:.1%}"
                    )
                ).rename(
                    columns={
                        "symbol": "Asset",
                        "regime": "Regime",
                        "direction_accuracy": "Direction accuracy",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        st.divider()
        st.subheader("ML Feature Importance (Macro Drivers)")
        st.caption(
            "Gini feature importances from tree-based return models. "
            "Shows which top macro indicators "
            "(Shiller CAPE, Yield Curve, VIX, NFCI) drive predictions per asset."
        )
        glossary.glossary_tooltip("cape")
        feat_syms = sorted(metrics.loc[metrics["model"] == "return_gb", "symbol"].dropna().unique())
        if feat_syms:
            spy_idx = feat_syms.index("SPY") if "SPY" in feat_syms else 0
            sel_fi_sym = st.selectbox("Feature importance asset", feat_syms, index=spy_idx)
            chart_wrapper(charts.feature_importance_chart(metrics, symbol=sel_fi_sym, height=300))
    else:
        st.info("No return forecasts available. Run `mmi ml` to generate predictions.")
