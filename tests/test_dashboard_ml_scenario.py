"""Tests for dashboard/tabs/ml_scenario.py — the ML-tab scenario analysis section.

The Streamlit rendering layer (st.* calls) is exercised end-to-end by `make app-smoke`;
here we smoke-test the render function with faked st.* calls and a fake chart_wrapper,
plus data-shape assertions on the scenario simulation chart itself.
"""

import pandas as pd
import pytest
from dashboard.components import charts
from dashboard.tabs import ml_scenario

_TAB_COLS = ["symbol", "model", "horizon", "predicted_return", "daily_mu", "as_of"]


def _fc(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_TAB_COLS)


def _return_fc(rows: list[tuple]) -> pd.DataFrame:
    """A `return_forecast_table`-shaped table (the column contract scenario uses)."""
    return pd.DataFrame(
        rows, columns=["symbol", "as_of", "horizon", "predicted_return", "daily_mu"]
    )


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TestModuleSurface:
    def test_module_imports(self):
        import importlib

        module = importlib.import_module("dashboard.tabs.ml_scenario")
        assert callable(module.render_ml_scenario_tab)

    def test_render_fn_exposed(self):
        assert callable(ml_scenario.render_ml_scenario_tab)


class TestRenderSmoke:
    @pytest.fixture(autouse=True)
    def _fake_st(self, monkeypatch):
        """Route every st.* call the section makes to in-memory fakes."""
        self.seen = {"sliders": {}, "charts": [], "subheaders": [], "dividers": 0}

        def _columns(n):
            return [_FakeColumn() for _ in range(n)]

        def _slider(label, *_args, **kwargs):
            value = kwargs.get("value", 0)
            self.seen["sliders"][label] = value
            return value

        monkeypatch.setattr(
            ml_scenario.st,
            "divider",
            lambda: self.seen.update(dividers=self.seen["dividers"] + 1),
        )
        monkeypatch.setattr(
            ml_scenario.st, "subheader", lambda t: self.seen["subheaders"].append(t)
        )
        monkeypatch.setattr(ml_scenario.st, "caption", lambda _t: None)
        monkeypatch.setattr(ml_scenario.st, "columns", _columns)
        monkeypatch.setattr(ml_scenario.st, "slider", _slider)

    def test_renders_section_and_chart(self, monkeypatch):
        fc = _return_fc(
            [
                ("TLT", "2026-08-01", 20, -0.02, 0.001),
                ("SPY", "2026-08-01", 20, 0.01, 0.001),
            ]
        )
        monkeypatch.setattr(ml_scenario.data, "ml_forecast", lambda: _fc([]))
        monkeypatch.setattr(ml_scenario.charts, "return_forecast_table", lambda _df: fc)
        captured = {}
        monkeypatch.setattr(
            ml_scenario.charts,
            "scenario_simulation_chart",
            lambda _fc_df, delta_rate_bps=0.0, delta_vix=0.0, height=300: captured.update(
                rate=delta_rate_bps, vix=delta_vix, height=height
            ),
        )

        calls = []
        ml_scenario.render_ml_scenario_tab(calls.append)

        assert self.seen["subheaders"] == ["⚡ Interactive Macro Scenario Stress-Tester"]
        assert self.seen["dividers"] == 1
        assert self.seen["sliders"] == {"Fed Funds Rate Shift (bps)": 0, "VIX Index Shift": 0}
        assert captured == {"rate": 0, "vix": 0, "height": 300}
        assert len(calls) == 1  # the fake chart_wrapper received the figure once

    def test_no_chart_without_forecasts(self, monkeypatch):
        monkeypatch.setattr(ml_scenario.data, "ml_forecast", lambda: _fc([]))
        monkeypatch.setattr(ml_scenario.charts, "return_forecast_table", lambda _df: _return_fc([]))
        monkeypatch.setattr(ml_scenario.charts, "scenario_simulation_chart", lambda *_a, **_k: None)

        calls = []
        ml_scenario.render_ml_scenario_tab(calls.append)

        assert calls == []  # nothing to stress-test → no chart, no crash


class TestScenarioChartShape:
    def test_grouped_baseline_vs_shocked_traces(self):
        fc = _return_fc(
            [
                ("SPY", "2026-08-01", 20, 0.01, 0.001),
                ("TLT", "2026-08-01", 20, -0.02, 0.001),
                ("GLD", "2026-08-01", 20, 0.005, 0.001),
            ]
        )
        fig = charts.scenario_simulation_chart(fc, delta_rate_bps=25, delta_vix=2, height=300)

        names = [tr.name for tr in fig.data]
        assert names == ["Baseline Forecast", "Simulated Macro Shock"]
        assert list(fig.data[0].x) == ["SPY", "TLT", "GLD"]

    def test_shock_shift_direction(self):
        fc = _return_fc([("SPY", "2026-08-01", 20, 0.01, 0.001)])
        fig = charts.scenario_simulation_chart(fc, delta_rate_bps=25, delta_vix=2)
        baseline = fig.data[0].y[0]
        shocked = fig.data[1].y[0]
        # SPY has negative rate & vix sensitivities → a hike/spike lowers the forecast.
        assert shocked < baseline
        assert shocked == pytest.approx(baseline + (25 / 100.0) * -0.05 + 2 * -0.008)

    def test_empty_forecast_renders_placeholder(self):
        fig = charts.scenario_simulation_chart(_return_fc([]))
        assert len(fig.data) == 0
        assert "Scenario Simulator (no forecast data)" in (fig.layout.title.text or "")


class TestDataShape:
    def test_scenario_uses_ml_forecast_contract(self, monkeypatch):
        """The section feeds the return-forecast column contract into the chart."""
        raw = _fc(
            [
                ("SPY", "return_gb", 20, 0.01, 0.001, "2026-08-01"),
                ("SPY", "vol_model", 20, 0.5, 0.0, "2026-08-01"),
                ("TLT", "return_lgbm", 20, -0.02, 0.001, "2026-08-01"),
            ]
        )
        monkeypatch.setattr(ml_scenario.data, "ml_forecast", lambda: raw)
        table = charts.return_forecast_table(raw)

        assert {"symbol", "horizon", "predicted_return", "daily_mu"} <= set(table.columns)
        assert set(table["symbol"]) == {"SPY", "TLT"}  # non-return models filtered out
        assert table["predicted_return"].iloc[0] == pytest.approx(0.01)  # sorted desc

        fig = charts.scenario_simulation_chart(table, delta_rate_bps=0, delta_vix=0)
        assert len(fig.data) == 2
        assert list(fig.data[0].x) == ["SPY", "TLT"]
