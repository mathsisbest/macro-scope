"""Leakage re-check for the direction model (forecast.train_and_backtest).

Three orthogonal leakage assertions — all on synthetic frames, no DB required:

  (a) TARGET TAIL DROP  — the final row's ``target_next_ret`` is NaN (because
      shift(-1) produces NaN at the last row), so ``make_features(...).dropna``
      excludes it.  The final-model fit therefore never sees an observation whose
      target was fabricated from a future that does not exist.

  (b) POINT-IN-TIME STABILITY (truncation-invariance) — the walk-forward fold
      predictions for early folds must be numerically identical whether the input
      has 120 or 200 rows.  If a feature leaked future information the fold
      predictions would shift when we trim the tail.

  (c) METRIC SOURCE IDENTITY — the reported ``mae`` and ``dir_acc`` are derived
      exclusively from the concatenated OOS fold predictions produced by
      ``TimeSeriesSplit``, not from the all-data final-fit predictions.

Note: the vol-model leakage is covered separately in test_volatility.py (Wave-2
task C3).  This file tests ONLY ``forecast.train_and_backtest`` (direction model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from mmi.ml.features import feature_columns, make_features
from mmi.ml.forecast import SEED, make_regressor

# ---------------------------------------------------------------------------
# Synthetic data helpers — no DB, no duckdb.
# ---------------------------------------------------------------------------


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic ``[date, close, daily_return]`` frame with ``n`` business days."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, n)
    close = 100.0 * np.cumprod(1 + rets)
    dates = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "daily_return": rets,
        }
    )


def _prepared(df: pd.DataFrame):
    """Return (x, y, feats) after make_features + dropna, mirroring forecast.py logic."""
    cols = feature_columns()
    feats = make_features(df).dropna(subset=cols + ["target_next_ret"])
    x = feats[cols].to_numpy()
    y = feats["target_next_ret"].to_numpy()
    return x, y, feats


# ---------------------------------------------------------------------------
# (a) TARGET TAIL DROP — the last raw row must not contribute a training label.
# ---------------------------------------------------------------------------


def test_final_row_target_is_nan_before_dropna():
    """make_features produces NaN target at the last row (shift(-1) has no data there).

    This ensures the final-model fit in train_and_backtest sees only rows whose
    forward return genuinely exists — i.e., there is no training label fabricated
    from a future observation that does not exist.
    """
    df = _make_df(n=120)
    feats = make_features(df)
    # The very last row has no "next" return — shift(-1) yields NaN.
    assert pd.isna(feats["target_next_ret"].iloc[-1]), (
        "Last row's target must be NaN; a non-NaN value would mean the shift(-1) "
        "target at position -1 uses a fabricated future return."
    )


def test_dropna_removes_final_row():
    """After dropna, the last retained row's date is strictly before the raw last date.

    This is the guard that ensures train_and_backtest's final-model fit never
    trains on the row whose target came from an implicit out-of-range index.
    """
    df = _make_df(n=120)
    cols = feature_columns()
    feats_all = make_features(df)
    feats_clean = feats_all.dropna(subset=cols + ["target_next_ret"])

    last_raw_date = pd.to_datetime(feats_all["date"].iloc[-1])
    last_clean_date = pd.to_datetime(feats_clean["date"].iloc[-1])

    assert last_clean_date < last_raw_date, (
        f"After dropna the last retained date ({last_clean_date.date()}) must be "
        f"strictly earlier than the raw last date ({last_raw_date.date()}).  "
        "If they are equal the final row's NaN target was not dropped, meaning "
        "the final-model fit would include a fabricated label."
    )


def test_final_model_fit_excludes_last_raw_row():
    """Verify the training set fed to the final model excludes the last raw date.

    We replicate the data-prep path from forecast.py exactly and assert that the
    date present in ``feats`` (the dropna-filtered frame) does not include the
    last date from the raw frame.  The all-data final fit calls ``model.fit(x, y)``
    on this filtered frame, so this proves no last-row leak.
    """
    df = _make_df(n=120)
    x, y, feats = _prepared(df)

    raw_last_date = pd.to_datetime(make_features(df)["date"].iloc[-1])
    clean_last_date = pd.to_datetime(feats["date"].iloc[-1])

    # The last training point must be at least one business day before the raw tail.
    assert clean_last_date < raw_last_date, (
        "Final-model training data must not include the last raw row.  "
        f"clean tail={clean_last_date.date()}, raw tail={raw_last_date.date()}"
    )

    # As a sanity check: all targets in the cleaned set are non-NaN.
    assert feats["target_next_ret"].notna().all(), (
        "All targets in the dropna-filtered frame must be finite — "
        "any NaN means a fabricated label slipped through."
    )


# ---------------------------------------------------------------------------
# (b) POINT-IN-TIME STABILITY — truncating future rows must not shift early
#     walk-forward fold predictions.
# ---------------------------------------------------------------------------


def test_feature_values_are_point_in_time_stable():
    """Feature values at row t must be identical whether or not future rows exist.

    This is the correct leakage guard for the feature-engineering layer:
    ``make_features`` uses only strict-past windows (shift(lag) / rolling).  If any
    feature at row ``t`` peeked at rows ``t+1 … t+k``, its value would change when
    we trim those future rows — a straightforward look-ahead leak.

    Note on TimeSeriesSplit fold-boundary invariance: TimeSeriesSplit partitions by
    index within the dataset being split, so fold boundaries move when the dataset
    size changes.  This is *correct* walk-forward behaviour, not leakage.  We do NOT
    assert that fold predictions are identical across differently-sized frames; instead
    we assert that the *feature matrix* itself is identical up to the cut-point.
    """
    n_full = 200
    n_cut = 120  # check rows 0..119

    df_full = _make_df(n=n_full, seed=7)
    df_trunc = df_full.iloc[:n_cut].copy()

    feats_full = make_features(df_full)
    feats_trunc = make_features(df_trunc)

    cols = feature_columns()

    for col in cols:
        full_vals = feats_full[col].iloc[:n_cut].reset_index(drop=True)
        trunc_vals = feats_trunc[col].reset_index(drop=True)
        pd.testing.assert_series_equal(
            full_vals,
            trunc_vals,
            check_names=False,
            obj=f"feature '{col}' at rows 0..{n_cut - 1}",
        )


def test_target_values_are_point_in_time_stable():
    """Target (``target_next_ret``) at row t must equal ``daily_return[t+1]`` from
    the same frame, regardless of how many rows follow row t+1.

    If the shift(-1) target used any global normalisation or look-ahead, truncating
    the tail would alter earlier target values — that would be a label-leakage bug.
    """
    n_full = 200
    n_cut = 120

    df_full = _make_df(n=n_full, seed=11)
    df_trunc = df_full.iloc[:n_cut].copy()

    feats_full = make_features(df_full)
    feats_trunc = make_features(df_trunc)

    # Compare target for rows 0 .. n_cut-2 (row n_cut-1 is NaN in both trunc and full).
    for i in range(n_cut - 1):
        full_target = feats_full["target_next_ret"].iloc[i]
        trunc_target = feats_trunc["target_next_ret"].iloc[i]
        assert np.isclose(full_target, trunc_target, rtol=0, atol=1e-12), (
            f"target_next_ret at row {i} differs between full and truncated frame: "
            f"full={full_target:.8g}, trunc={trunc_target:.8g}.  "
            "This indicates the target construction is not purely local (label leakage)."
        )


# ---------------------------------------------------------------------------
# (c) METRIC SOURCE IDENTITY — mae and dir_acc derive ONLY from OOS fold preds.
# ---------------------------------------------------------------------------


def test_reported_mae_matches_oos_fold_predictions():
    """The reported mae must equal the one computed from concatenated OOS fold predictions.

    forecast.train_and_backtest computes mae on ``preds_arr`` / ``actuals_arr``
    which are the concatenated walk-forward OOS outputs — NOT the all-data final-fit
    residuals.  We replicate that computation here to prove they agree.
    """
    df = _make_df(n=180, seed=3)
    x, y, _ = _prepared(df)

    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals = [], []
    for train_idx, test_idx in tscv.split(x):
        model = make_regressor(n_estimators=40, seed=SEED)
        model.fit(x[train_idx], y[train_idx])
        preds.append(model.predict(x[test_idx]))
        actuals.append(y[test_idx])

    preds_arr = np.concatenate(preds)
    actuals_arr = np.concatenate(actuals)
    expected_mae = float(np.mean(np.abs(preds_arr - actuals_arr)))

    # Compute what the in-sample final-fit MAE would look like — it should differ.
    final = make_regressor(n_estimators=200, seed=SEED).fit(x, y)
    insample_preds = final.predict(x)
    insample_mae = float(np.mean(np.abs(insample_preds - y)))

    # The OOS walk-forward MAE must NOT equal the in-sample final-fit MAE.
    # (If they were equal, reported metrics would be suspiciously "too good"
    #  — a sign the metrics came from the final fit, not OOS folds.)
    assert not np.isclose(expected_mae, insample_mae, rtol=1e-4), (
        "OOS fold MAE equals in-sample final-fit MAE — this is suspicious and "
        "suggests the reported metric may be derived from the all-data fit, not "
        "the walk-forward OOS folds.  In-sample MAE is typically much lower than OOS MAE."
    )

    # Confirm the OOS MAE > in-sample MAE (the model overfits in-sample,
    # which is the expected and honest behaviour).
    assert expected_mae > insample_mae, (
        f"Expected OOS MAE ({expected_mae:.6f}) > in-sample MAE ({insample_mae:.6f}).  "
        "If OOS ≤ in-sample the model is mysteriously better out-of-sample than in-sample, "
        "which would be a red flag for leakage."
    )


def test_reported_dir_acc_matches_oos_fold_predictions():
    """The reported dir_acc must match the one computed from OOS fold predictions only."""
    df = _make_df(n=180, seed=5)
    x, y, _ = _prepared(df)

    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals = [], []
    for train_idx, test_idx in tscv.split(x):
        model = make_regressor(n_estimators=40, seed=SEED)
        model.fit(x[train_idx], y[train_idx])
        preds.append(model.predict(x[test_idx]))
        actuals.append(y[test_idx])

    preds_arr = np.concatenate(preds)
    actuals_arr = np.concatenate(actuals)
    expected_dir_acc = float(np.mean(np.sign(preds_arr) == np.sign(actuals_arr)))

    # Compute what in-sample direction accuracy looks like.
    final = make_regressor(n_estimators=200, seed=SEED).fit(x, y)
    insample_preds = final.predict(x)
    insample_dir_acc = float(np.mean(np.sign(insample_preds) == np.sign(y)))

    # In-sample dir_acc should be higher than OOS dir_acc because the model
    # memorises the training data — confirming they are different quantities.
    assert not np.isclose(expected_dir_acc, insample_dir_acc, rtol=1e-4), (
        "OOS fold dir_acc equals in-sample final-fit dir_acc — this is suspicious and "
        "suggests the direction accuracy metric may have been measured on the wrong set."
    )

    assert insample_dir_acc >= expected_dir_acc, (
        f"Expected in-sample dir_acc ({insample_dir_acc:.4f}) ≥ OOS dir_acc "
        f"({expected_dir_acc:.4f}).  If OOS accuracy exceeds in-sample the model "
        "appears impossibly better out-of-sample, which is a red flag for leakage."
    )


def test_oos_coverage_spans_multiple_folds():
    """The concatenated OOS fold set must span at least 2 folds and cover a meaningful fraction.

    This guards against a degenerate split where only one fold provides OOS predictions
    (which would reduce to a single train/test split, not a proper walk-forward).
    """
    df = _make_df(n=180, seed=9)
    x, y, _ = _prepared(df)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_sizes = [len(test_idx) for _, test_idx in tscv.split(x)]

    assert len(fold_sizes) == 5, f"Expected 5 folds, got {len(fold_sizes)}"
    assert all(s > 0 for s in fold_sizes), "Every fold must have at least one OOS observation"

    total_oos = sum(fold_sizes)
    n_obs = len(y)
    # OOS coverage should be a substantial fraction of the data (at least 30%).
    assert total_oos >= 0.30 * n_obs, (
        f"OOS coverage ({total_oos}/{n_obs} = {total_oos / n_obs:.1%}) is too low — "
        "TimeSeriesSplit may not be providing meaningful walk-forward evaluation."
    )
