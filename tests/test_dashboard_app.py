"""Test the pure-logic helpers extracted from dashboard/app.py.

The Streamlit rendering layer (st.* calls) is not tested here — it is exercised by the
`make app-smoke` end-to-end runner. This file covers the testable data-wiring functions.
"""

import pandas as pd

from dashboard.components.utils import provenance_badge, sidebar_status


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
