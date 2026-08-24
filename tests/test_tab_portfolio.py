"""Tests for dashboard/tabs/portfolio.py — the Portfolio tab rendering and gross vs net returns."""

from __future__ import annotations

import pandas as pd
import pytest
from dashboard import data
from dashboard.tabs import portfolio


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _sample_portfolio_returns(n_days: int = 50) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    strategies = ["equal_weight", "inverse_vol", "risk_parity", "sixty_forty"]
    rows = []
    for strat in strategies:
        cum = 0.0
        for d in dates:
            ret = 0.0005
            cum = (1 + cum) * (1 + ret) - 1
            rows.append(
                {
                    "strategy": strat,
                    "date": d,
                    "daily_return": ret,
                    "cumulative_return": cum,
                    "drawdown": 0.0,
                    "rolling_sharpe_252": 1.2,
                }
            )
    return pd.DataFrame(rows)


def _sample_attribution() -> pd.DataFrame:
    rows = []
    for strat in ["equal_weight", "inverse_vol", "risk_parity", "sixty_forty"]:
        rows.extend(
            [
                {
                    "strategy": strat,
                    "symbol": "SPY",
                    "contribution_to_return": 0.08,
                    "contribution_to_risk": 0.6,
                    "strategy_gross_return": 0.12,
                },
                {
                    "strategy": strat,
                    "symbol": "TLT",
                    "contribution_to_return": 0.04,
                    "contribution_to_risk": 0.4,
                    "strategy_gross_return": 0.12,
                },
                {
                    "strategy": strat,
                    "symbol": "(costs)",
                    "contribution_to_return": -0.015,
                    "contribution_to_risk": 0.0,
                    "strategy_gross_return": 0.12,
                },
            ]
        )
    return pd.DataFrame(rows)


class TestRenderPortfolioTabSmoke:
    @pytest.fixture(autouse=True)
    def _fake_st(self, monkeypatch):
        self.seen = {
            "markdown": [],
            "caption": [],
            "info": [],
            "dataframe": [],
            "selectbox": {},
            "radio": {},
            "expanders": [],
        }

        def _columns(n):
            if isinstance(n, int):
                return [_FakeColumn() for _ in range(n)]
            return [_FakeColumn() for _ in range(len(n))]

        class _Expander:
            def __init__(self, title):
                self.title = title

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(portfolio.st, "columns", _columns)
        monkeypatch.setattr(
            portfolio.st, "expander", lambda t, **k: _Expander(t)
        )
        monkeypatch.setattr(
            portfolio.st, "markdown", lambda m, **k: self.seen["markdown"].append(m)
        )
        monkeypatch.setattr(
            portfolio.st, "caption", lambda c, **k: self.seen["caption"].append(c)
        )
        monkeypatch.setattr(
            portfolio.st, "info", lambda i, **k: self.seen["info"].append(i)
        )
        monkeypatch.setattr(
            portfolio.st, "dataframe", lambda d, **k: self.seen["dataframe"].append(d)
        )
        monkeypatch.setattr(
            portfolio.st,
            "selectbox",
            lambda label, opts, **k: opts[0] if opts else None,
        )
        monkeypatch.setattr(
            portfolio.st,
            "radio",
            lambda label, opts, **k: opts[0] if opts else None,
        )

    def test_renders_with_mocked_data(self, monkeypatch):
        pf = _sample_portfolio_returns(30)
        attr = _sample_attribution()
        monkeypatch.setattr(data, "portfolio_windows", lambda: ["ex_btc_2002"])
        monkeypatch.setattr(data, "portfolio_returns", lambda *a, **k: pf)
        monkeypatch.setattr(data, "portfolio_strategy_stats", lambda *a, **k: pd.DataFrame())
        monkeypatch.setattr(data, "portfolio_strategy_pairs", lambda *a, **k: pd.DataFrame())
        monkeypatch.setattr(data, "portfolio_attribution", lambda *a, **k: attr)
        monkeypatch.setattr(data, "portfolio_regime_performance", lambda *a, **k: pd.DataFrame())
        monkeypatch.setattr(data, "portfolio_ml_gate", lambda *a, **k: pd.DataFrame())
        monkeypatch.setattr(data, "portfolio_btc_effect", lambda: pd.DataFrame())
        monkeypatch.setattr(data, "regimes", lambda sym: pd.DataFrame())

        charts_rendered = []
        portfolio.render_portfolio_tab(None, charts_rendered.append)

        assert len(charts_rendered) >= 2  # cumulative chart + gross/net chart + attribution
        assert any(
            "Gross vs Net" in str(getattr(c.layout.title, "text", ""))
            for c in charts_rendered
            if hasattr(c, "layout") and hasattr(c.layout, "title")
        )

    def test_handles_no_windows_gracefully(self, monkeypatch):
        monkeypatch.setattr(data, "portfolio_windows", lambda: [])
        charts_rendered = []
        portfolio.render_portfolio_tab(None, charts_rendered.append)
        assert len(charts_rendered) == 0
        assert len(self.seen["info"]) > 0
