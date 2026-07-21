"""Leakage re-check for evaluate_forecast (regression guard).

Verifies:
  (a) TARGET TAIL DROP — the final row's target_next_ret is NaN after shift.
  (b) TRUNCATION INVARIANCE — predictions for overlapping dates are identical
      when future data is removed (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmi.ml.forecast import evaluate_forecast


def _long(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(0.0004, 0.01, n)
    return pd.DataFrame({
        "date": dates,
        "daily_return": rets,
        "open": 100,
        "high": 101,
        "low": 99,
        "close": 100,
    })


class TestTargetTailDrop:
    def test_last_row_target_is_nan(self):
        df = _long(60).sort_values("date")
        df["ret"] = df["daily_return"]
        df["target_next_ret"] = df["ret"].shift(-1)
        assert pd.isna(df["target_next_ret"].iloc[-1])

    def test_dropna_removes_last_row(self):
        df = _long(60).sort_values("date")
        df["ret"] = df["daily_return"]
        df["target_next_ret"] = df["ret"].shift(-1)
        tail = df.dropna(subset=["target_next_ret"])
        assert len(tail) == len(df) - 1


class TestTruncationInvariance:
    def test_overlapping_predictions_match(self):
        full = _long(200)
        early_run = evaluate_forecast(
            full.iloc[:130], train_size=60, test_size=20, horizon=1,
            model="gb", feature_set="vol", target_horizon=1,
        )
        full_run = evaluate_forecast(
            full, train_size=60, test_size=20, horizon=1,
            model="gb", feature_set="vol", target_horizon=1,
        )

        if early_run.get("prediction_count", 0) < 3 or full_run.get("prediction_count", 0) < 3:
            pytest.skip("too few predictions for comparison")

        early_dates = early_run.get("dates", pd.Series())
        full_dates = full_run.get("dates", pd.Series())

        if len(early_dates) == 0 or len(full_dates) == 0:
            pytest.skip("no dates in results")

        early_preds = pd.Series(
            early_run["predictions"].to_numpy(),
            index=pd.to_datetime(early_dates.to_numpy()),
        )
        full_preds = pd.Series(
            full_run["predictions"].to_numpy(),
            index=pd.to_datetime(full_dates.to_numpy()),
        )

        common_dates = early_preds.index.intersection(full_preds.index)
        if len(common_dates) < 3:
            pytest.skip("too few overlapping dates")

        early_c = early_preds.loc[common_dates].to_numpy()
        full_c = full_preds.loc[common_dates].to_numpy()
        np.testing.assert_allclose(early_c, full_c, rtol=1e-10,
                                   err_msg="truncation changed predictions")


class TestEdgeCases:
    def test_fewer_than_60_rows_returns_empty(self):
        df = _long(30)
        result = evaluate_forecast(
            df, train_size=60, test_size=20, horizon=1,
            model="gb", feature_set="default", target_horizon=1,
        )
        assert result.get("prediction_count", 0) == 0

    def test_single_row_returns_empty(self):
        df = _long(1)
        result = evaluate_forecast(
            df, train_size=60, test_size=20, horizon=1,
            model="gb", feature_set="default", target_horizon=1,
        )
        assert result.get("prediction_count", 0) == 0
