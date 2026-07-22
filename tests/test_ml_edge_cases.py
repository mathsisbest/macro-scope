"""Edge-case tests for evaluate_forecast.

Covers graceful degradation on small / constant / single-row inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmi.ml.forecast import evaluate_forecast


def _series(rows: int, const: bool = False) -> pd.DataFrame:
    np.random.seed(42)
    ret = np.zeros(rows) if const else np.random.normal(0.0004, 0.01, rows)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=rows),
            "daily_return": ret,
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
        }
    )


class TestEdgeCases:
    def test_fewer_than_60_rows_returns_empty(self):
        result = evaluate_forecast(
            _series(30),
            train_size=60,
            test_size=20,
            horizon=1,
            model="gb",
            feature_set="default",
        )
        assert result.get("prediction_count", 0) == 0

    def test_single_row_returns_empty(self):
        result = evaluate_forecast(
            _series(1),
            train_size=60,
            test_size=20,
            horizon=1,
            model="gb",
            feature_set="default",
        )
        assert result.get("prediction_count", 0) == 0

    def test_constant_returns_produces_valid_result(self):
        result = evaluate_forecast(
            _series(200, const=True),
            train_size=60,
            test_size=20,
            horizon=1,
            model="gb",
            feature_set="vol",
        )
        if result.get("prediction_count", 0) > 0:
            assert "ic" in result

    def test_many_rows(self):
        result = evaluate_forecast(
            _series(500),
            train_size=100,
            test_size=20,
            horizon=1,
            model="gb",
            feature_set="default",
        )
        assert result.get("prediction_count", 0) > 0
        assert result.get("ic") is not None

    def test_gb_model_returns_result(self):
        result = evaluate_forecast(
            _series(200),
            train_size=60,
            test_size=20,
            horizon=1,
            model="gb",
            feature_set="default",
        )
        assert result.get("model") == "gb"

    def test_hyperparameter_auto_tuning(self):
        result = evaluate_forecast(
            _series(200),
            train_size=60,
            test_size=20,
            horizon=1,
            model="gb",
            feature_set="default",
            tune_hyperparameters=True,
            single_split=True,
        )
        assert result.get("prediction_count", 0) > 0
