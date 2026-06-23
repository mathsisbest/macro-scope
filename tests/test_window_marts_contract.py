"""The window enum (single source of truth) must not drift from the dbt accepted_values.

Phase D stamps a ``window_id`` on every portfolio mart and constrains it in ``_marts.yml`` with an
``accepted_values`` test. That YAML list is hand-maintained, so this test asserts it matches
``mmi.portfolio.windows.WINDOWS`` exactly — if someone adds a window to the enum but forgets the
YAML (or vice versa), CI fails here instead of a window silently slipping past the dbt contract.
"""

from pathlib import Path

import yaml

from mmi.portfolio import windows

_MARTS_YML = Path(__file__).resolve().parents[1] / "transform" / "models" / "marts" / "_marts.yml"
_PORTFOLIO_MARTS = {
    "fct_portfolio_returns",
    "fct_portfolio_strategy_stats",
    "fct_portfolio_strategy_pairs",
    "fct_performance_attribution",
    "fct_portfolio_regime_performance",
    "fct_portfolio_ml_gate",
}


def _accepted_values(column: dict) -> set[str] | None:
    for test in column.get("tests", []):
        if isinstance(test, dict) and "accepted_values" in test:
            return set(test["accepted_values"]["values"])
    return None


def test_every_portfolio_mart_constrains_window_id_to_the_enum():
    models = yaml.safe_load(_MARTS_YML.read_text())["models"]
    by_name = {m["name"]: m for m in models}
    expected = set(windows.WINDOWS)

    checked = set()
    for name in _PORTFOLIO_MARTS:
        assert name in by_name, f"{name} missing from _marts.yml"
        cols = {c["name"]: c for c in by_name[name]["columns"]}
        assert "window_id" in cols, f"{name} has no window_id column block"
        values = _accepted_values(cols["window_id"])
        assert values == expected, f"{name}.window_id accepted_values {values} != enum {expected}"
        checked.add(name)
    assert checked == _PORTFOLIO_MARTS
