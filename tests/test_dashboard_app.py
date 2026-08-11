"""Test the pure-logic helpers extracted from dashboard/app.py.

The Streamlit rendering layer (st.* calls) is not tested here — it is exercised by the
`make app-smoke` end-to-end runner. This file covers the testable data-wiring functions.
"""

import pandas as pd
from dashboard.components import kpi
from dashboard.components.utils import provenance_badge, sidebar_status


class TestYieldCurveSpreadPick:
    """The KPI row's spread-column selection: honest label per chosen column."""

    def _mm(self, **columns) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-01", periods=3)
        return pd.DataFrame({"date": dates, **columns})

    def test_prefers_10y_3m_with_label_when_present(self):
        mm = self._mm(yield_curve_10y_3m=[0.5, 0.4, 0.3], yield_curve_10y_2y=[0.2, 0.1, 0.0])
        assert kpi.yield_curve_spread_pick(mm) == ("yield_curve_10y_3m", "10Y−3M spread")

    def test_falls_back_to_10y_2y_with_honest_label(self):
        # 3M column absent (older snapshot) — the tile must say "10Y−2Y spread", not lie.
        mm = self._mm(yield_curve_10y_2y=[0.2, 0.1, 0.0])
        assert kpi.yield_curve_spread_pick(mm) == ("yield_curve_10y_2y", "10Y−2Y spread")

    def test_falls_back_when_3m_column_all_nan(self):
        mm = self._mm(yield_curve_10y_3m=[None, None, None], yield_curve_10y_2y=[0.2, 0.1, 0.0])
        assert kpi.yield_curve_spread_pick(mm) == ("yield_curve_10y_2y", "10Y−2Y spread")

    def test_none_when_no_spread_data(self):
        assert kpi.yield_curve_spread_pick(self._mm(yield_curve_10y_2y=[None] * 3)) is None
        no_cols = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=3)})
        assert kpi.yield_curve_spread_pick(no_cols) is None

    def test_none_on_empty_or_malformed_input(self):
        assert kpi.yield_curve_spread_pick(pd.DataFrame()) is None
        assert kpi.yield_curve_spread_pick(None) is None


class TestProvenanceBadge:
    def test_live_data(self):
        badge = provenance_badge("2026-06-30", False)
        assert "Data as of **2026-06-30**" in badge
        assert "live data" in badge

    def test_sample_data(self):
        badge = provenance_badge("2026-06-30", True)
        assert "sample data" in badge

    def test_unrecorded(self):
        badge = provenance_badge("2026-06-30", None)
        assert "mixed/unrecorded" in badge

    def test_no_as_of(self):
        assert provenance_badge(None, None) == ""

    def test_no_as_of_with_is_sample_false(self):
        badge = provenance_badge(None, False)
        assert "live data" in badge


class TestSidebarStatus:
    def test_hidden_when_runs_present(self):
        runs = pd.DataFrame({"source": ["yahoo"], "status": ["ok"]})
        assert sidebar_status(None, None, runs) == ""

    def test_sample_message(self):
        assert "Sample" in sidebar_status(True, None, pd.DataFrame())

    def test_live_snapshot_message(self):
        assert "snapshot" in sidebar_status(False, None, pd.DataFrame())

    def test_mixed_provenance(self):
        assert "Mixed" in sidebar_status(None, "2026-06-30", pd.DataFrame())

    def test_no_data_message(self):
        assert "No data yet" in sidebar_status(None, None, pd.DataFrame())
