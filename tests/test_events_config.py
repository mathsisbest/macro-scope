"""Unit tests for config/events.yml loading and chart annotations (R9).

Tests:
1. Dynamic loading of `config/events.yml` via `load_events()`.
2. Fallback to default events on missing/malformed file or unreadable path.
3. Settings integration with `MMI_EVENTS_PATH` environment variable alias.
4. Dashboard data accessor `events()` with TypedDict shape and defaults.
5. Chart annotation helpers: `get_chart_events()` and `add_event_annotations()`.
6. Time-series chart functions (`price_chart`, `vol_chart`, `rebased_performance_chart`,
   `macro_chart`, `yield_curve_chart`, `portfolio_cumulative_chart`) supporting `events=True`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from dashboard import data
from dashboard.components import charts
from dashboard.snapshot_boot import configure_dashboard_env

from mmi.settings import DEFAULT_EVENTS, Settings, load_events


def test_load_events_from_default_config():
    """Verify loading the real config/events.yml from repository root."""
    res = load_events()
    assert "events" in res
    events = res["events"]
    assert len(events) >= 10

    labels = [e["label"] for e in events]
    assert "COVID Market Low" in labels
    assert "Fed First Rate Hike" in labels
    assert "SVB Collapse" in labels
    assert "Lehman Collapse" in labels

    # Dates and categories verified
    covid_event = next(e for e in events if e["label"] == "COVID Market Low")
    assert covid_event["date"] == "2020-03-23"
    assert covid_event["category"] == "market_shock"

    fed_event = next(e for e in events if e["label"] == "Fed First Rate Hike")
    assert fed_event["date"] == "2022-03-16"
    assert fed_event["category"] == "monetary_policy"


def test_load_events_fallback_on_missing_file(tmp_path: Path):
    """When path does not exist, load_events falls back to DEFAULT_EVENTS."""
    missing_path = tmp_path / "non_existent_events.yml"
    res = load_events(missing_path)
    assert res == {"events": DEFAULT_EVENTS}


def test_load_events_fallback_on_invalid_yaml(tmp_path: Path):
    """When path contains invalid content, load_events safely falls back."""
    bad_path = tmp_path / "bad_events.yml"
    bad_path.write_text("not a valid yaml: : : [", encoding="utf-8")
    res = load_events(bad_path)
    assert res == {"events": DEFAULT_EVENTS}


def test_load_events_custom_valid_file(tmp_path: Path):
    """When a custom valid events.yml is provided, it is parsed accurately."""
    custom_path = tmp_path / "custom_events.yml"
    custom_path.write_text(
        "events:\n"
        "  - date: '2025-01-01'\n"
        "    label: 'Custom Milestone'\n"
        "    description: 'A test event'\n"
        "    category: 'custom'\n",
        encoding="utf-8",
    )
    res = load_events(custom_path)
    assert len(res["events"]) == 1
    assert res["events"][0]["label"] == "Custom Milestone"
    assert res["events"][0]["date"] == "2025-01-01"


def test_settings_events_path_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify MMI_EVENTS_PATH environment variable overrides events_path."""
    custom_events = tmp_path / "env_events.yml"
    custom_events.write_text("events: []\n", encoding="utf-8")
    monkeypatch.setenv("MMI_EVENTS_PATH", str(custom_events))
    s = Settings(_env_file=None)
    assert s.events_path == custom_events


def test_configure_dashboard_env_pins_events_path(tmp_path: Path):
    """configure_dashboard_env sets MMI_EVENTS_PATH default to repo_root/config/events.yml."""
    env: dict[str, str] = {}
    configure_dashboard_env(env, tmp_path)
    assert env["MMI_EVENTS_PATH"] == str(tmp_path / "config" / "events.yml")

    # Operator override is preserved
    env2 = {"MMI_EVENTS_PATH": "/custom/events.yml"}
    configure_dashboard_env(env2, tmp_path)
    assert env2["MMI_EVENTS_PATH"] == "/custom/events.yml"


def test_dashboard_data_events_accessor():
    """Verify data.events() returns formatted list of EventItems."""
    evs = data.events()
    assert isinstance(evs, list)
    assert len(evs) > 0
    first = evs[0]
    assert {"date", "label", "description", "category"}.issubset(first.keys())


def test_get_chart_events_date_filtering():
    """Verify get_chart_events clips events to [start, end]."""
    mock_events = [
        {"date": "2020-01-01", "label": "E1", "category": "market_shock"},
        {"date": "2021-06-01", "label": "E2", "category": "crisis"},
        {"date": "2023-01-01", "label": "E3", "category": "monetary_policy"},
    ]
    # No filter
    all_evs = charts.get_chart_events(events_list=mock_events)
    assert len(all_evs) == 3

    # Start filter
    filtered_start = charts.get_chart_events(start="2021-01-01", events_list=mock_events)
    assert [e["label"] for e in filtered_start] == ["E2", "E3"]

    # End filter
    filtered_end = charts.get_chart_events(end="2022-01-01", events_list=mock_events)
    assert [e["label"] for e in filtered_end] == ["E1", "E2"]

    # Both start and end
    bounded = charts.get_chart_events(start="2021-01-01", end="2022-01-01", events_list=mock_events)
    assert [e["label"] for e in bounded] == ["E2"]


def test_add_event_annotations_adds_lines_and_labels():
    """Verify add_event_annotations injects vlines and text annotations."""
    fig = charts.go.Figure()
    mock_events = [
        {"date": "2020-03-23", "label": "COVID Low", "category": "market_shock"},
        {"date": "2022-03-16", "label": "Fed Hike", "category": "monetary_policy"},
    ]
    charts.add_event_annotations(fig, events_list=mock_events)

    # Check vline shapes
    vlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(vlines) == 2
    assert all(s.line.dash == "dot" for s in vlines)

    # Check text annotations
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["COVID Low", "Fed Hike"]
    for a in fig.layout.annotations:
        assert a.y == 1.0
        assert a.yref == "paper"
        assert a.yanchor == "bottom"


def test_price_chart_with_events():
    """Verify price_chart integrates event annotations when events=True."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-12-31"]),
            "close": [100.0, 110.0, 120.0],
        }
    )
    mock_events = [
        {"date": "2020-03-23", "label": "COVID Market Low", "category": "market_shock"},
        {"date": "2023-03-10", "label": "SVB Collapse", "category": "crisis"},  # out of date range
    ]
    fig = charts.price_chart(df, symbol="SPY", events=True, events_list=mock_events)
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["COVID Market Low"]
    assert len([s for s in fig.layout.shapes if s.type == "line"]) == 1


def test_vol_chart_with_events():
    """Verify vol_chart integrates event annotations when events=True."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-12-31"]),
            "vol_20d": [0.15, 0.35, 0.20],
        }
    )
    mock_events = [
        {"date": "2020-03-23", "label": "COVID Market Low", "category": "market_shock"}
    ]
    fig = charts.vol_chart(df, symbol="SPY", events=True, events_list=mock_events)
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["COVID Market Low"]


def test_macro_chart_with_events():
    """Verify macro_chart integrates event annotations when events=True."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-01", "2022-06-01", "2022-12-31"]),
            "value": [0.25, 1.50, 4.25],
        }
    )
    mock_events = [
        {"date": "2022-03-16", "label": "Fed First Rate Hike", "category": "monetary_policy"}
    ]
    fig = charts.macro_chart(df, label="Fed Funds", events=True, events_list=mock_events)
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["Fed First Rate Hike"]


def test_yield_curve_chart_with_events():
    """Verify yield_curve_chart integrates event annotations when events=True."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-01", "2022-06-01", "2022-12-31"]),
            "yield_curve_10y_2y": [0.8, -0.2, -0.5],
        }
    )
    mock_events = [
        {"date": "2022-06-13", "label": "10Y-2Y Inversion", "category": "crisis"}
    ]
    fig = charts.yield_curve_chart(df, events=True, events_list=mock_events)
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["10Y-2Y Inversion"]


def test_portfolio_cumulative_chart_with_events():
    """Verify portfolio_cumulative_chart integrates event annotations when events=True."""
    df = pd.DataFrame(
        {
            "strategy": ["equal_weight"] * 3,
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-12-31"]),
            "cumulative_return": [0.0, 0.05, 0.15],
        }
    )
    mock_events = [
        {"date": "2020-03-23", "label": "COVID Low", "category": "market_shock"}
    ]
    fig = charts.portfolio_cumulative_chart(df, events=True, events_list=mock_events)
    labels = [a.text for a in fig.layout.annotations]
    assert "COVID Low" in labels
