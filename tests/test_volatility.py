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
    _HAR_COLS,
    _VOL_FLOOR,
    MODEL_TAG,
    _ewma_vol,
    _fit_predict_har,
    _make_targets,
    _qlike,
    _walk_forward_ewma_baseline,
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


def test_ewma_baseline_no_forward_dependence() -> None:
    """The walk-forward EWMA baseline for a fold must not depend on rows AFTER that fold.

    Mutating gk-vol rows beyond a fold's last test point must leave that fold's baseline
    predictions byte-identical — that is the honest-OOS contract the skill gate relies on.
    As a guard against a vacuous test, mutating a row WITHIN the window must change them.
    """
    from sklearn.model_selection import TimeSeriesSplit

    rng = np.random.default_rng(11)
    n = 120
    gk = np.abs(rng.normal(0.01, 0.003, n))

    # Use a real TimeSeriesSplit fold — the same splitter train_and_backtest_vol uses.
    splits = list(TimeSeriesSplit(n_splits=5).split(np.zeros((n, 1))))
    _, test_idx = splits[2]  # a middle fold, so rows exist both before and after it
    last = int(test_idx.max())
    assert last < n - 1, "fold must leave rows after the test window to mutate"

    base = _walk_forward_ewma_baseline(gk, test_idx)

    # Mutate every row strictly AFTER the test window — the baseline must be unchanged.
    gk_future = gk.copy()
    gk_future[last + 1 :] = gk_future[last + 1 :] * 7.0 + 0.5
    base_future = _walk_forward_ewma_baseline(gk_future, test_idx)
    np.testing.assert_array_equal(
        base, base_future, err_msg="baseline leaked future rows (forward dependence)"
    )

    # Sanity (non-vacuity): mutating a row WITHIN the test window DOES move the baseline.
    gk_inside = gk.copy()
    gk_inside[test_idx[0]] *= 3.0
    base_inside = _walk_forward_ewma_baseline(gk_inside, test_idx)
    assert not np.allclose(base, base_inside), "baseline should react to in-window changes"


def test_qlike_positive() -> None:
    """QLIKE must be >= 0 for reasonable actuals and predictions."""
    actuals = np.abs(np.random.default_rng(1).normal(0.01, 0.002, 50))
    preds = np.abs(np.random.default_rng(2).normal(0.01, 0.002, 50))
    assert _qlike(actuals, preds) >= 0.0, "QLIKE must be non-negative"


def test_qlike_zero_for_perfect_predictions() -> None:
    """QLIKE is approximately 0 when predictions equal actuals."""
    v = np.abs(np.random.default_rng(3).normal(0.01, 0.002, 50))
    assert pytest.approx(_qlike(v, v), abs=1e-10) == 0.0


def test_qlike_bounded_when_realized_vol_near_zero() -> None:
    """A near-zero realised-vol day must NOT blow QLIKE up — the under-specified-baseline bug.

    QLIKE divides predicted by realised variance, so without the ``_VOL_FLOOR`` a single flat
    Garman-Klass day (realised ≈ 0) sends pred²/realised² to ~1e10 and dominates the whole mean
    (that is what made ``baseline_qlike`` ≈ 1121 and ``qlike_skill_ratio`` a free pass).  With the
    floor the loss stays finite and modest.  The floored ratio at the bad point is
    ``(0.01/_VOL_FLOOR)² = 25`` → contribution ``25 - ln 25 - 1 ≈ 20.8`` spread over 50 rows ≈ 0.42.
    """
    actuals = np.full(50, 0.01)
    actuals[0] = 1e-8  # a degenerate near-flat day
    preds = np.full(50, 0.01)

    q = _qlike(actuals, preds)
    assert np.isfinite(q)
    assert q < 1.0, f"QLIKE should stay bounded with the vol floor, got {q}"


def test_har_extrapolates_above_training_range() -> None:
    """A linear log-HAR must extrapolate ABOVE the vol it saw in training.

    This is the property a random forest lacks — and the reason calm-train/crisis-test folds
    collapsed to a negative OOS R² under the old borrowed-forest estimator.  We fit on a calm
    regime (~1% daily vol) where the target tracks the cascade level, then predict a crisis row
    whose cascade sits well above anything in training; a forest would clip at its training max,
    the HAR projects past it.
    """
    rng = np.random.default_rng(0)
    x_train = rng.uniform(0.008, 0.012, size=(200, len(_HAR_COLS)))
    x_train_log = np.log(np.clip(x_train, _VOL_FLOOR, None))
    y_train = x_train.mean(axis=1)  # vol level ~0.01, linear in the cascade
    x_crisis_log = np.log(np.clip(np.full((1, len(_HAR_COLS)), 0.05), _VOL_FLOOR, None))

    pred = _fit_predict_har(x_train_log, y_train, x_crisis_log)[0]
    assert pred > y_train.max(), (
        f"log-HAR should extrapolate above the training max ({y_train.max():.4f}); got {pred:.4f}"
    )


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
# Test 6: Total row count after run_ml (direction: 10 rows, vol: 11 rows = 21 total for SPY)
# ---------------------------------------------------------------------------


def test_model_metrics_row_count(con) -> None:  # noqa: ANN001
    """After run_ml(['SPY']), model_metrics must have exactly 21 rows for SPY.

    The 400-day sample data is large enough to carve a locked holdout for BOTH models
    (~20% tail, dev still well above _MIN_OBS=60), so the holdout_* rows ARE present.

    direction (random_forest): mae, baseline_mae, dir_acc, baseline_dir_acc, n_obs,
                                mae_skill_ratio, dir_acc_edge          = 7 CV rows
                                + holdout_dir_acc, holdout_baseline_dir_acc,
                                  holdout_n_obs                         = 3 holdout rows = 10
    vol (rv_har):               oos_r2, qlike, baseline_qlike, qlike_skill_ratio,
                                n_folds, folds_passed, n_obs           = 7 CV rows
                                + holdout_oos_r2, holdout_qlike, holdout_qlike_skill_ratio,
                                  holdout_n_obs                         = 4 holdout rows = 11
    Total: 21 rows
    """
    from mmi.ml.pipeline import run_ml

    _seed_con(con)
    run_ml(con, symbols=["SPY"])

    count = con.execute("select count(*) from marts.model_metrics where symbol = 'SPY'").fetchone()[
        0
    ]

    assert count == 21, f"Expected 21 model_metrics rows for SPY, got {count}"


# ---------------------------------------------------------------------------
# Test 7: Locked holdout (honest extra OOS readout — reported, NOT gated)
# ---------------------------------------------------------------------------


def test_vol_holdout_rows_persisted(con) -> None:  # noqa: ANN001
    """run_ml on the 400-day sample carves a holdout; the 4 holdout_* rv_har rows exist."""
    from mmi.ml.pipeline import run_ml

    _seed_con(con)
    run_ml(con, symbols=["SPY"])

    rows = con.execute(
        "select metric, value from marts.model_metrics where model = ? and symbol = 'SPY'",
        [MODEL_TAG],
    ).fetchall()
    metric_map = {r[0]: r[1] for r in rows}

    expected = {
        "holdout_oos_r2",
        "holdout_qlike",
        "holdout_qlike_skill_ratio",
        "holdout_n_obs",
    }
    assert expected.issubset(metric_map.keys()), (
        f"missing holdout rv_har rows: {expected - metric_map.keys()}"
    )
    # On 375 feature-valid vol rows the holdout slice is 20% = 75 rows, minus the last
    # (_HORIZON - 1) = 4 rows whose forward window falls outside the slice -> 71 scored rows.
    assert metric_map["holdout_n_obs"] == 71.0
    assert metric_map["holdout_qlike"] >= 0.0


def test_vol_holdout_disjoint_from_cv_dev(con) -> None:  # noqa: ANN001
    """The holdout is the TAIL and is disjoint from every dev CV train/test fold (row-level).

    This replicates the feature-carve + split path of train_and_backtest_vol: the holdout
    rows are the last `hold` feature-valid rows, and NONE of their indices appear in any
    train or test fold of the dev-only TimeSeriesSplit.  (The stronger LABEL-disjointness is
    proven in test_vol_dev_cv_metrics_ignore_holdout_period.)
    """
    from sklearn.model_selection import TimeSeriesSplit

    from mmi.ml.features import feature_columns, make_features
    from mmi.ml.holdout import split_indices
    from mmi.ml.volatility import _MIN_OBS

    _seed_con(con)
    df = con.execute(
        "select date, open, high, low, close, daily_return from marts.fct_asset_daily "
        "where symbol = 'SPY' order by date"
    ).df()
    feats = make_features(df, feature_set="vol")
    vol_cols = feature_columns(feature_set="vol")
    # Carve on FEATURE-valid rows (the model builds the target per-slice, so we split first).
    feat_valid = feats.dropna(subset=vol_cols).reset_index(drop=True)
    n_feat = len(feat_valid)

    dev_end, hold = split_indices(n_feat, min_dev=_MIN_OBS)
    assert hold > 0, "sample data should be large enough to carve a holdout"

    holdout_idx = set(range(dev_end, n_feat))
    assert holdout_idx == set(range(n_feat - hold, n_feat)), "holdout must be exactly the tail"

    # The CV runs on dev only — collect every index it touches and assert disjointness.
    dev_idx_seen: set[int] = set()
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(range(dev_end)):
        dev_idx_seen.update(train_idx.tolist())
        dev_idx_seen.update(test_idx.tolist())

    assert dev_idx_seen.isdisjoint(holdout_idx), "leakage: a dev CV fold touched a holdout row"
    assert max(dev_idx_seen) < dev_end, "dev CV must never index into the holdout tail"


def test_vol_dev_cv_inputs_ignore_holdout_period(con) -> None:  # noqa: ANN001
    """P1-A LABEL-leak guard: NO dev CV input (feature OR label) may depend on a holdout value.

    The forward target mean(gk[t+1..t+horizon]) is built PER-SLICE, so dev's last `horizon`
    rows have incomplete forward windows and drop out before the CV — no dev label reads a
    holdout-period gk.  We prove it directly on the data the CV consumes: replicate the model's
    feature-carve + per-slice target build, then mutate every holdout-period OHLC row and
    assert the dev (x, y, gk) arrays are BYTE-IDENTICAL.  Under the old full-series target the
    dev tail labels would shift, so y would differ — this test would fail, making the fix
    self-enforcing.

    We assert on the dev INPUT arrays rather than the fitted metrics on purpose: byte-identical
    inputs is the exact, confound-free statement of label-disjointness, independent of which
    estimator consumes them (the model is now a deterministic log-OLS HAR; asserting on inputs
    held even back when it was a non-bit-reproducible n_jobs=-1 forest).

    Non-vacuity: mutating a DEV row instead DOES change the dev inputs.
    """
    from mmi.ml.features import feature_columns, make_features
    from mmi.ml.holdout import split_indices
    from mmi.ml.volatility import _HORIZON, _MIN_OBS, _make_targets

    _seed_con(con)
    base = con.execute(
        "select symbol, date, open, high, low, close, daily_return "
        "from marts.fct_asset_daily where symbol = 'SPY' order by date"
    ).df()

    vol_cols = feature_columns(feature_set="vol")

    def _dev_inputs(frame) -> tuple:  # noqa: ANN001
        """Replicate the model's dev-slice prep: carve features, then build the target per slice."""
        feats = make_features(frame, feature_set="vol")
        feat_valid = feats.dropna(subset=vol_cols).reset_index(drop=True)
        dev_end, hold = split_indices(len(feat_valid), min_dev=_MIN_OBS)
        assert hold > 0, "sample data should carve a holdout"
        f = feat_valid.iloc[:dev_end].copy()
        f["target_rv"] = _make_targets(f["gk_vol"], horizon=_HORIZON)
        f = f.dropna(subset=["target_rv"])
        return (
            f[vol_cols].to_numpy(),
            f["target_rv"].to_numpy(),
            f["gk_vol"].to_numpy(),
            dev_end,
        )

    x0, y0, g0, dev_end = _dev_inputs(base)
    # Map the holdout's first feature-valid date back to a raw-row position.
    holdout_first_date = (
        make_features(base, feature_set="vol")
        .dropna(subset=vol_cols)["date"]
        .reset_index(drop=True)
        .iloc[dev_end]
    )
    raw_holdout_start = int((base["date"] >= holdout_first_date).idxmax())

    # Helper: widen the daily high/low RANGE at the given rows so Garman-Klass vol genuinely
    # changes (scaling all of OHLC by a constant leaves the high/low ratio — and thus gk — fixed,
    # which would make the poison a no-op).
    def _widen_range(frame, row_slice):  # noqa: ANN001, ANN202
        f = frame.copy()
        idx = f.index[row_slice]
        f.loc[idx, "high"] = f.loc[idx, "high"] * 1.5
        f.loc[idx, "low"] = f.loc[idx, "low"] * 0.5
        return f

    # (1) Poison the HOLDOUT period: widen the range for every raw row at/after the holdout start.
    poisoned = _widen_range(base, slice(raw_holdout_start, None))
    xp, yp, gp, _ = _dev_inputs(poisoned)
    assert np.array_equal(x0, xp), "LABEL LEAK: a dev FEATURE changed when only holdout rows moved"
    assert np.array_equal(y0, yp), (
        "LABEL LEAK: a dev TARGET changed when only HOLDOUT-period rows were mutated — a dev "
        "label is reading a holdout-period gk value (the P1-A bug)."
    )
    assert np.array_equal(g0, gp), "LABEL LEAK: a dev gk value changed when only holdout rows moved"

    # (2) Non-vacuity: mutating a DEV row DOES change the dev inputs.
    dev_row = max(0, raw_holdout_start - 30)  # safely inside the dev period
    dev_poisoned = _widen_range(base, slice(dev_row, dev_row + 1))
    xd, _, _, _ = _dev_inputs(dev_poisoned)
    assert not np.array_equal(x0, xd), (
        "mutating a DEV row left the dev features unchanged — the leak test is vacuous"
    )


def test_vol_holdout_skipped_on_small_data() -> None:
    """When the dev portion would fall below _MIN_OBS, the holdout is skipped (no keys, no crash).

    We use 90 OHLC rows -> 71 feature-valid rows: holdout_size=floor(0.2*71)=14 -> dev would be
    57 < _MIN_OBS=60 -> SKIP.  With the holdout skipped, the CV runs on all 71 feature-valid
    rows (minus the last _HORIZON dropped to the forward window, ~66 trainable >= 60), so we
    get the normal metrics but NO holdout_* keys.
    """
    con = duckdb.connect(":memory:")
    init_schemas(con)

    n = 90
    df = _make_ohlc_df(n=n).assign(symbol="SPY")
    cols = [
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
    con.register("_d", df[cols])
    con.execute("CREATE OR REPLACE TABLE marts.fct_asset_daily AS SELECT * FROM _d")
    con.unregister("_d")

    metrics, fc = train_and_backtest_vol(con, "SPY")
    assert metrics, "CV should still run on a >= _MIN_OBS series"
    assert fc is not None
    # No holdout was carved -> no holdout_* keys.
    holdout_keys = [k for k in metrics if k.startswith("holdout_")]
    assert holdout_keys == [], f"expected NO holdout keys on small data, got {holdout_keys}"
    # n_obs equals the full valid count (holdout skipped means dev == full series).
    assert metrics["n_obs"] > 0


def test_vol_holdout_not_in_gate_metrics(con) -> None:  # noqa: ANN001
    """The skill gate must be unaffected by the holdout — it reads only the CV metric rows.

    skill_verdict() filters on the five gate metric NAMES; the holdout_* rows share the model
    tag but are never among those names, so the verdict is identical whether or not they exist.
    """
    from mmi.ml.pipeline import run_ml
    from mmi.ml.skill_gate import skill_verdict

    _seed_con(con)
    run_ml(con, symbols=["SPY"])
    full = con.execute("select * from marts.model_metrics").df()

    verdict_with_holdout = skill_verdict(full, "SPY")
    # Drop every holdout_* row and re-run: the verdict must be byte-identical.
    no_holdout = full[~full["metric"].str.startswith("holdout_")]
    verdict_without = skill_verdict(no_holdout, "SPY")

    assert verdict_with_holdout == verdict_without, (
        "skill_verdict changed when holdout rows were removed — the gate is NOT supposed to "
        "see the holdout (it is reported, not gated)"
    )


def test_holdout_ewma_baseline_is_warm_started() -> None:
    """The holdout EWMA baseline must inherit the dev EWMA state (warm), not restart cold.

    A cold start re-seeds the λ=0.94 (~16-day memory) recursion from the holdout's first GK value
    with no prior history; the fix carries the dev observations forward by concatenating gk_dev
    ahead of gk_hold and reading the EWMA at the holdout offsets.  With the dev period at a
    different vol level the two disagree on the early holdout predictions — and the cold version
    biases the *reported* holdout baseline (hence ``holdout_qlike_skill_ratio``).  This guards the
    exact mechanism ``train_and_backtest_vol`` uses.
    """
    gk_dev = np.full(200, 0.02)  # long, warm dev history at a HIGH vol level
    gk_hold = np.full(40, 0.01)  # holdout slice at a LOW vol level

    # WARM (the fix): concatenate dev ahead of the holdout, read at the holdout offsets.
    gk_warm = np.concatenate([gk_dev, gk_hold])
    hold_idx = np.arange(len(gk_dev), len(gk_warm))
    warm = _walk_forward_ewma_baseline(gk_warm, hold_idx)

    # COLD (the previous behaviour): recurse over the holdout slice alone.
    cold = _walk_forward_ewma_baseline(gk_hold, np.arange(len(gk_hold)))

    # adjust=False EWMA seeds from the first input, so a cold start == the holdout's first value.
    assert cold[0] == pytest.approx(gk_hold[0]), "cold start seeds from the holdout's first value"
    # The warm first prediction must reflect the dev memory, not restart from gk_hold[0].
    assert warm[0] != pytest.approx(cold[0]), "warm start must differ from a cold recompute"
    assert warm[0] > cold[0], "warm start should carry the high-vol dev level into the holdout"

    # The warm series must equal a single full-history EWMA recompute read at the holdout offsets.
    full_history = _ewma_vol(pd.Series(gk_warm)).to_numpy()[hold_idx]
    np.testing.assert_allclose(warm, full_history)


def test_train_and_backtest_vol_uses_warm_holdout_baseline(con, monkeypatch) -> None:  # noqa: ANN001
    """Call-site guard: the holdout baseline must be handed dev history + holdout (warm), not the
    holdout slice alone (cold).

    Spying on ``_walk_forward_ewma_baseline``, the CV folds pass ``gk_dev`` (length ``n_obs``) and
    the holdout call passes ``gk_dev`` concatenated with ``gk_hold`` (strictly longer).  A
    cold-start regression would hand the holdout call a series exactly ``holdout_n_obs`` long, so
    no call would exceed ``n_obs`` and this assertion would fail.
    """
    import mmi.ml.volatility as vol

    captured: list[tuple[int, np.ndarray]] = []
    orig = vol._walk_forward_ewma_baseline

    def _spy(gk, test_idx, lam=vol._EWMA_LAMBDA):  # noqa: ANN001, ANN202
        captured.append((len(gk), np.asarray(test_idx).copy()))
        return orig(gk, test_idx, lam=lam)

    monkeypatch.setattr(vol, "_walk_forward_ewma_baseline", _spy)

    _seed_con(con)
    metrics, _ = vol.train_and_backtest_vol(con, "SPY")

    assert "holdout_n_obs" in metrics, "sample data should carve a holdout"
    n_dev = int(metrics["n_obs"])
    n_hold = int(metrics["holdout_n_obs"])

    # CV calls pass gk_dev (length n_obs); the holdout call passes gk_dev + gk_hold (longer).
    holdout_calls = [(gk_len, idx) for gk_len, idx in captured if gk_len > n_dev]
    assert len(holdout_calls) == 1, (
        f"expected exactly one warm (dev+holdout) baseline call longer than n_obs={n_dev}, got "
        f"{len(holdout_calls)} — a cold-start regression would size the holdout gk at "
        f"holdout_n_obs={n_hold}"
    )
    gk_len, test_idx = holdout_calls[0]
    assert gk_len == n_dev + n_hold, "holdout baseline gk must be dev history + holdout slice"
    # The holdout is read at offsets PAST the dev history (warm seed), scoring n_hold rows.
    assert test_idx[0] == n_dev, "holdout baseline must read past the dev history (warm offset)"
    assert len(test_idx) == n_hold
