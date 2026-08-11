"""ML Scenario Analysis Tab Module.

Interactive what-if stress-testing of the ML 20-day return forecasts against
macro shocks (Fed Funds rate shift, VIX spike). Rendered on the ML tab, after
the forecast section — the scenario simulator is meaningless without the
forecasts it perturbs.
"""

from __future__ import annotations

import streamlit as st
from dashboard import data
from dashboard.components import charts


def render_ml_scenario_tab(chart_wrapper) -> None:
    """Render the interactive macro scenario stress-tester for the ML forecasts."""
    st.divider()
    st.subheader("⚡ Interactive Macro Scenario Stress-Tester")
    st.caption(
        "Simulate custom macro shocks (Fed Funds rate shift, VIX spike) "
        "on 20-day predicted asset returns."
    )
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        rate_shock = st.slider(
            "Fed Funds Rate Shift (bps)", -200, +200, 0, step=25, key="ml_scenario_rate_shock"
        )
    with sim_col2:
        vix_shock = st.slider("VIX Index Shift", -10, +20, 0, step=1, key="ml_scenario_vix_shock")

    sim_fc = charts.return_forecast_table(data.ml_forecast())
    if not sim_fc.empty:
        chart_wrapper(
            charts.scenario_simulation_chart(
                sim_fc, delta_rate_bps=rate_shock, delta_vix=vix_shock, height=300
            )
        )
