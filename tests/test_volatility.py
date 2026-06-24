"""Tests for the HAR realized-volatility model (task C3).

Covers:
  1. Leakage-free: features at row t use only data <= t; target is strictly forward.
  2. Persistence/EWMA baseline is computed correctly (positive, finite).
  3. Small-sample safety: fewer than _MIN_OBS rows returns ({}, None) without raising.
  4. Full run persists rv_har metric rows + ml_forecast row to marts.
  5. Direction-model honest secondary rows (mae_skill_ratio, dir_acc_edge) are persisted.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from mmi import sampledata, transform_fallback
from mmi.ml.volatility import (
    MODEL_TAG,
    _ewma_vol,
    _make_targets,
    _qlike,
    train_and_backtest_vol,
)
from mmi.utils.db import init_schemas

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlc_df(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLC + daily_return dataframe (enough rows for a full walk-forward)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.012, n))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    daily_return = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "daily_return": daily_return,
            "source": "sample",
            "asset_class": "equities",
            "volume": 0,
            "vol_20d": 0.0,
            "ma_50": 100.0,
        }
    )


def _in_memory_con_with_data(n: int = 120) -> duckdb.DuckDBPyConnection:
    """In-memory DB with fct_asset_daily seeded with `n` rows for SPY."""
    con = duckdb.connect(":memory:")
    init_schemas(con)
    sampledata.seed(con)
    transform_fallback.build_marts(con)
    return con


# ---------------------------------------------------------------------------
# Test 1: Leakage-free — forward targets are strictly forward
# ---------------------------------------------------------------------------


def test_make_targets_strictly_forward() -> None:
    """_make_targets at row t must use only rows t+1..t+horizon (no look-back into itself)."""
    n = 30
    gk = pd.Series(np.arange(1.0, n + 1))  # deterministic monotone series

    targets = _make_targets(gk, horizon=5)

    # Check a mid-series row: e.g. row 10 (0-indexed) should be mean(gk[11..15]) = mean(12..16)
    # gk is 1-indexed values: gk[10]=11.0, gk[11]=12.0,...gk[15]=16.0
    # target[10] = mean(gk[11], gk[12], gk[13], gk[14], gk[15]) = mean(12,13,14,15,16)=14.0
    expected = (12.0 + 13.0 + 14.0 + 15.0 + 16.0) / 5.0
    assert pytest.approx(targets.iloc[10], rel=1e-9) == expected, (
        f"target[10] should be {expected}, got {targets.iloc[10]}"
    )


def test_vol_features_no_future_leakage_in_training() -> None:
    """Replacing future OHLC rows with NaN must not change earlier vol features."""
    from mmi.ml.features import feature_columns, make_features

    n, cut = 60, 30
    df = _make_ohlc_df(n=n, seed=99)

    # Poison rows from cut onwards
    df_poisoned = df.copy()
    df_poisoned.loc[cut:, ["open", "high", "low", "close", "daily_return"]] = float("nan")

    clean = make_features(df.iloc[:cut].copy(), feature_set="vol")
    poisoned = make_features(df_poisoned.copy(), feature_set="vol")

    vol_cols = feature_columns(feature_set="vol")
    for col in vol_cols:
        pd.testing.assert_series_equal(
            clean[col].reset_index(drop=True),
            poisoned[col].iloc[:cut].reset_index(drop=True),
            check_names=False,
            obj=f"leakage check col={col}",
        )


# ---------------------------------------------------------------------------
# Test 2: Persistence/EWMA baseline is positive and finite
# ---------------------------------------------------------------------------


def test_ewma_vol_positive_finite() -> None:
    """EWMA of GK vol must be positive and finite for a well-formed series."""
    gk = pd.Series(np.abs(np.random.default_rng(0).normal(0.01, 0.002, 100)))
    result = _ewma_vol(gk)
    assert result.notna().all(), "EWMA should contain no NaN"
    assert (result > 0).all(), "EWMA should be positive"
    assert np.isfinite(result).all(), "EWMA should be finite"


def test_qlike_positive() -> None:
    """QLIKE must be >= 0 for reasonable actuals and predictions."""
    actuals = np.abs(np.random.default_rng(1).normal(0.01, 0.002, 50))
    preds = np.abs(np.random.default_rng(2).normal(0.01, 0.002, 50))
    assert _qlike(actuals, preds) >= 0.0, "QLIKE must be non-negative"


def test_qlike_zero_for_perfect_predictions() -> None:
    """QLIKE is approximately 0 when predictions equal actuals."""
    v = np.abs(np.random.default_rng(3).normal(0.01, 0.002, 50))
    assert pytest.approx(_qlike(v, v), abs=1e-10) == 0.0


# ---------------------------------------------------------------------------
# Test 3: Small-sample safety — fewer than _MIN_OBS rows returns ({}, None)
# ---------------------------------------------------------------------------


def test_small_sample_returns_empty_no_crash() -> None:
    """With fewer than _MIN_OBS observations, train_and_backtest_vol must not raise."""
    con = duckdb.connect(":memory:")
    init_schemas(con)

    # Insert only 10 rows for SPY — way below the minimum
    n = 10
    df = _make_ohlc_df(n=n).assign(symbol="SPY")
    df_small = df[
        [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "daily_return",
            "source",
            "asset_class",
            "volume",
            "vol_20d",
            "ma_50",
        ]
    ]
    con.register("_tiny", df_small)
    con.execute("CREATE OR REPLACE TABLE marts.fct_asset_daily AS SELECT * FROM _tiny")
    con.unregister("_tiny")

    metrics, fc = train_and_backtest_vol(con, "SPY")
    assert metrics == {}, "Expected empty metrics on small-sample skip"
    assert fc is None, "Expected None forecast on small-sample skip"


# ---------------------------------------------------------------------------
# Test 4: Full run persists rv_har metric rows + ml_forecast row
# ---------------------------------------------------------------------------


def _seed_con(con) -> None:
    """Seed synthetic data + build marts into the given connection."""
    sampledata.seed(con)
    transform_fallback.build_marts(con)


def test_rv_har_rows_persisted_to_marts(con) -> None:  # noqa: ANN001
    """After run_ml, marts.model_metrics must contain rv_har rows with expected metric names."""
    from mmi.ml.pipeline import run_ml

    _seed_con(con)
    run_ml(con, symbols=["SPY"])

    rows = con.execute(
        "select metric, value from marts.model_metrics where model = ? and symbol = 'SPY'",
        [MODEL_TAG],
    ).fetchall()

    metric_names = {r[0] for r in rows}
    expected_metrics = {
        "oos_r2",
        "qlike",
        "baseline_qlike",
        "qlike_skill_ratio",
        "n_folds",
        "folds_passed",
        "n_obs",
    }
    missing = expected_metrics - metric_names
    assert not missing, f"Missing rv_har metric rows: {missing}"


def test_rv_har_forecast_in_ml_forecast(con) -> None:  # noqa: ANN001
    """After run_ml, marts.ml_forecast must contain a row with model='rv_har'."""
    from mmi.ml.pipeline import run_ml

    _seed_con(con)
    run_ml(con, symbols=["SPY"])

    rows = con.execute(
        "select symbol, as_of, model from marts.ml_forecast where model = ?",
        [MODEL_TAG],
    ).fetchall()

    assert len(rows) == 1, f"Expected 1 rv_har forecast row, got {len(rows)}"
    assert rows[0][0] == "SPY"
    assert rows[0][2] == MODEL_TAG


def test_rv_har_n_obs_is_positive(con) -> None:  # noqa: ANN001
    """rv_har n_obs metric must be positive."""
    from mmi.ml.pipeline import run_ml

    _seed_con(con)
    run_ml(con, symbols=["SPY"])

    row = con.execute(
        "select value from marts.model_metrics "
        "where model = ? and symbol = 'SPY' and metric = 'n_obs'",
        [MODEL_TAG],
    ).fetchone()

    assert row is not None, "n_obs row must exist"
    assert row[0] > 0, f"n_obs must be positive, got {row[0]}"


# ---------------------------------------------------------------------------
# Test 5: Direction-model honest secondary rows (C4: mae_skill_ratio, dir_acc_edge)
# ---------------------------------------------------------------------------


def test_direction_model_skill_rows_persisted(con) -> None:  # noqa: ANN001
    """direction model metrics must include mae_skill_ratio + dir_acc_edge (random_forest)."""
    from mmi.ml.pipeline import run_ml

    _seed_con(con)
    run_ml(con, symbols=["SPY"])

    rows = con.execute(
        "select metric, value from marts.model_metrics "
        "where model = 'random_forest' and symbol = 'SPY'"
    ).fetchall()

    metric_names = {r[0] for r in rows}
    assert "mae_skill_ratio" in metric_names, "mae_skill_ratio row missing for random_forest"
    assert "dir_acc_edge" in metric_names, "dir_acc_edge row missing for random_forest"


def test_direction_model_mae_skill_ratio_formula(con) -> None:  # noqa: ANN001
    """mae_skill_ratio == mae / baseline_mae (cross-check the formula)."""
    from mmi.ml.pipeline import run_ml

    _seed_con(con)
    run_ml(con, symbols=["SPY"])

    def _get(metric):
        return con.execute(
            "select value from marts.model_metrics "
            "where model='random_forest' and symbol='SPY' and metric=?",
            [metric],
        ).fetchone()[0]

    mae = _get("mae")
    baseline_mae = _get("baseline_mae")
    mae_skill_ratio = _get("mae_skill_ratio")

    if baseline_mae > 1e-20:
        assert pytest.approx(mae_skill_ratio, rel=1e-6) == mae / baseline_mae


def test_direction_model_dir_acc_edge_formula(con) -> None:  # noqa: ANN001
    """dir_acc_edge == dir_acc - baseline_dir_acc."""
    from mmi.ml.pipeline import run_ml

    _seed_con(con)
    run_ml(con, symbols=["SPY"])

    def _get(metric):
        return con.execute(
            "select value from marts.model_metrics "
            "where model='random_forest' and symbol='SPY' and metric=?",
            [metric],
        ).fetchone()[0]

    dir_acc = _get("dir_acc")
    baseline_dir_acc = _get("baseline_dir_acc")
    dir_acc_edge = _get("dir_acc_edge")

    assert pytest.approx(dir_acc_edge, rel=1e-6) == dir_acc - baseline_dir_acc


# ---------------------------------------------------------------------------
# Test 6: Total row count after run_ml (direction: 7 rows, vol: 7 rows = 14 total for SPY)
# ---------------------------------------------------------------------------


def test_model_metrics_row_count(con) -> None:  # noqa: ANN001
    """After run_ml(['SPY']), model_metrics must have exactly 14 rows for SPY.

    direction (random_forest): mae, baseline_mae, dir_acc, baseline_dir_acc, n_obs,
                                mae_skill_ratio, dir_acc_edge = 7 rows
    vol (rv_har):               oos_r2, qlike, baseline_qlike, qlike_skill_ratio,
                                n_folds, folds_passed, n_obs = 7 rows
    Total: 14 rows
    """
    from mmi.ml.pipeline import run_ml

    _seed_con(con)
    run_ml(con, symbols=["SPY"])

    count = con.execute("select count(*) from marts.model_metrics where symbol = 'SPY'").fetchone()[
        0
    ]

    assert count == 14, f"Expected 14 model_metrics rows for SPY, got {count}"
