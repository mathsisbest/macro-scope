"""Guard against dead-module regressions in dashboard/tabs/.

Each tab's rendering is exercised end-to-end by `make app-smoke`; this cheap import guard
ensures the modules stay importable (and exposed to ruff/mypy) even if app.py ever stops
wiring one of them.
"""

import importlib

import pytest

TAB_MODULES = {
    "dashboard.tabs.digest": "render_digest_tab",
    "dashboard.tabs.markets": "render_markets_tab",
    "dashboard.tabs.macro": "render_macro_tab",
    "dashboard.tabs.ml_forecast": "render_ml_tab",
    "dashboard.tabs.ml_scenario": "render_ml_scenario_tab",
    "dashboard.tabs.portfolio": "render_portfolio_tab",
}


@pytest.mark.parametrize("module_name", sorted(TAB_MODULES))
def test_tab_module_imports_cleanly(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", sorted(TAB_MODULES))
def test_tab_module_exposes_render_fn(module_name: str) -> None:
    module = importlib.import_module(module_name)
    render_fn = TAB_MODULES[module_name]
    assert callable(getattr(module, render_fn)), f"{module_name} must expose {render_fn}()"
