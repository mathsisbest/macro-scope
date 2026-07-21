"""Tests for ML-tilted MVO portfolio strategy.

Covers compute_ml_mu_panel and compute_portfolio_returns with ML mu_panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mmi.portfolio.compute import compute_ml_mu_panel, compute_portfolio_returns


def _asset_daily(n: int = 252, symbols: list | None = None) -> pd.DataFrame:
    if symbols is None:
        symbols = ["SPY", "TLT"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    for sym in symbols:
        for d in dates:
            rows.append({
                "date": d,
                "symbol": sym,
                "daily_return": rng.normal(0.0004, 0.01),
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
            })
    return pd.DataFrame(rows)


class TestComputeMlMuPanel:
    def test_returns_mu_and_gate_dataframes(self):
        panel = _asset_daily(252)
        mu, gate = compute_ml_mu_panel(panel)
        assert isinstance(mu, pd.DataFrame)
        assert isinstance(gate, pd.DataFrame)

    def test_mu_has_expected_columns(self):
        panel = _asset_daily(252)
        mu, gate = compute_ml_mu_panel(panel)
        assert "mu" in mu.columns
        assert "symbol" in mu.columns
        assert "date" in mu.columns

    def test_gate_has_expected_columns(self):
        panel = _asset_daily(252)
        mu, gate = compute_ml_mu_panel(panel)
        assert "forecast_skill" in gate.columns
        assert "forecast_weight" in gate.columns

    def test_mu_values_are_finite(self):
        panel = _asset_daily(252)
        mu, gate = compute_ml_mu_panel(panel)
        assert mu.notna().all().all()
        mu_vals = pd.to_numeric(mu["mu"], errors="coerce")
        assert mu_vals.notna().all()


class TestComputePortfolioReturns:
    def test_includes_equal_weight_baseline(self):
        panel = _asset_daily(504, symbols=["SPY", "TLT", "GLD"])
        result = compute_portfolio_returns(panel)
        assert "equal_weight" in result["strategy"].to_numpy()

    def test_accepts_ml_mu_panel(self):
        panel = _asset_daily(504, symbols=["SPY", "TLT", "GLD"])
        mu, gate = compute_ml_mu_panel(panel)
        result = compute_portfolio_returns(panel, ml_mu_panel=mu)
        assert "equal_weight" in result["strategy"].to_numpy()

    def test_equal_weight_has_finite_returns(self):
        panel = _asset_daily(504, symbols=["SPY", "TLT", "GLD"])
        result = compute_portfolio_returns(panel)
        ew = result[result["strategy"] == "equal_weight"]
        assert ew["daily_return"].notna().all()
