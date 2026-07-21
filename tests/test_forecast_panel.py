"""Tests for ForecastPanel — multi-horizon ensemble forecast.

Covers run_forecast and run_universe methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mmi.ml.forecast_panel import ForecastBacktest


class _MockMacroDB:
    """Stub macro_db that returns a synthetic price series."""
    def __init__(self, n: int = 200):
        rng = np.random.default_rng(42)
        self.dates = pd.bdate_range("2020-01-01", periods=n)
        rets = rng.normal(0.0004, 0.01, n)
        self._df = pd.DataFrame({
            "date": self.dates,
            "daily_return": rets,
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
        })

    def prices_df(self, symbol: str) -> pd.DataFrame:
        return self._df

    macro_df = None
    asset_dfs = None


class TestForecastBacktest:
    def test_run_forecast_returns_expected_keys(self):
        mock_db = _MockMacroDB(200)
        bt = ForecastBacktest(["SPY"], mock_db)
        result = bt.run_forecast("SPY", horizons=(1, 5))
        assert "ic" in result
        assert "direction_accuracy" in result
        assert "prediction_count" in result

    def test_run_universe_returns_per_symbol_results(self):
        mock_db = _MockMacroDB(200)
        bt = ForecastBacktest(["SPY"], mock_db)
        results = bt.run_universe(progress=False)
        assert "SPY" in results
        assert "ic" in results["SPY"]
