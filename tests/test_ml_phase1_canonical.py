"""Unit tests for Phase 1 canonical splitters and evaluation metrics modules."""

import numpy as np
import pandas as pd
import pytest

from mmi.ml.metrics import (
    ForecastEvaluationResult,
    compute_directional_accuracy,
    compute_ic,
    compute_r2,
    compute_regime_directional_accuracy,
    compute_sharpe,
)
from mmi.ml.splitters import feasible_date_range, walk_forward_split


def test_walk_forward_split_standard():
    total_len = 100
    train_size = 50
    test_size = 10

    splits = list(walk_forward_split(total_len, train_size, test_size))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(train_idx) == 50
        assert len(test_idx) == 10
        assert train_idx[-1] + 1 == test_idx[0]


def test_walk_forward_split_single():
    total_len = 100
    train_size = 50
    test_size = 10

    splits = list(walk_forward_split(total_len, train_size, test_size, single_split=True))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert train_idx == list(range(0, 50))
    assert test_idx == list(range(50, 100))


def test_walk_forward_split_expanding():
    total_len = 80
    train_size = 40
    test_size = 20

    splits = list(walk_forward_split(total_len, train_size, test_size, use_all_train=True))
    assert len(splits) == 2
    # First fold: train 0..40, test 40..60
    assert splits[0][0] == list(range(0, 40))
    # Second fold: train 0..60 (expanding!), test 60..80
    assert splits[1][0] == list(range(0, 60))


def test_feasible_date_range():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates})

    first, last = feasible_date_range(df, train_size=5)
    assert first == dates[5]
    assert last == dates[9]

    # Insufficient length
    first_nat, last_nat = feasible_date_range(df, train_size=15)
    assert pd.isna(first_nat) and pd.isna(last_nat)


def test_compute_ic():
    yt = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
    yp = pd.Series([0.01, 0.015, -0.005, 0.025, -0.01])

    ic_val, ic_pval = compute_ic(yt, yp)
    assert ic_val > 0.95
    assert ic_pval < 0.05

    # Zero std edge case
    ic_zero, p_zero = compute_ic(pd.Series([0, 0, 0]), pd.Series([1, 2, 3]))
    assert ic_zero == 0.0
    assert p_zero == 1.0


def test_compute_directional_accuracy():
    yt = pd.Series([0.01, 0.02, -0.01, -0.03, 0.02])
    yp = pd.Series([0.02, 0.01, -0.02, 0.01, 0.01])  # 4 right, 1 wrong (-0.03 vs 0.01)

    metrics = compute_directional_accuracy(yt, yp)
    assert pytest.approx(metrics["direction_accuracy"], 0.01) == 0.8
    assert pytest.approx(metrics["positive_target_rate"], 0.01) == 0.6
    assert pytest.approx(metrics["baseline_direction_accuracy"], 0.01) == 0.6
    assert pytest.approx(metrics["direction_edge"], 0.01) == 0.2


def test_compute_sharpe():
    yt = pd.Series([0.01, 0.02, -0.01, -0.03, 0.02])
    yp = pd.Series([0.02, 0.01, -0.02, -0.01, 0.01])

    sharpe = compute_sharpe(yt, yp, target_horizon=1)
    assert not np.isnan(sharpe)
    assert sharpe > 0.0


def test_compute_r2():
    yt = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
    yp = pd.Series([0.01, 0.015, -0.005, 0.025, -0.01])

    r2_ic = compute_r2(yt, yp, method="ic_signed_sq")
    assert r2_ic > 0.9

    r2_reg = compute_r2(yt, yp, method="regression")
    assert r2_reg > 0.0


def test_forecast_evaluation_result_schema():
    res = ForecastEvaluationResult(ic=0.15, r2=0.0225, model="gb")
    assert res.ic == 0.15
    assert res["ic"] == 0.15
    assert res.get("r2") == 0.0225
    assert res.get("missing_key", "default") == "default"

    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["ic"] == 0.15
    assert d["r2"] == 0.0225
