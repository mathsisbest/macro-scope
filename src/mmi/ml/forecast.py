"""Next-day return forecasting with an honest walk-forward backtest.

Locked holdout (honest extra OOS readout)
-----------------------------------------
After feature construction the LAST ``holdout_size`` time-ordered rows are carved off as a
locked holdout (see :mod:`mmi.ml.holdout`).  The holdout is **never** in a CV training/test
fold; the walk-forward dir_acc/mae come from the DEV portion as before.  We then final-fit on
ALL of dev, predict the untouched holdout, and report ``holdout_dir_acc``,
``holdout_baseline_dir_acc`` (majority-class baseline on the holdout) and ``holdout_n_obs``.
These are **reported, not gated**, and are NEVER used to tune anything.  If carving would
leave fewer than ``_MIN_OBS`` dev rows the holdout is SKIPPED (no holdout_* keys; CV runs on
the full series as before).

Label-leak guard (P1-A): the target is ``ret.shift(-1)`` (horizon 1), so the LAST dev row's
label would otherwise read the FIRST holdout-period return.  We therefore carve the
feature-valid rows FIRST, then rebuild the target WITHIN the dev slice and WITHIN the holdout
slice — each slice's last row gets a NaN target (forward window outside the slice) and is
dropped, so no dev label depends on a holdout-period value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from mmi.ml.features import feature_columns, make_features

# MIN_OBS lives canonically in mmi.ml.holdout (the leaf module both models import), so the two
# models share ONE trainable-rows floor instead of duplicating the literal (P3 DRY).  Imported
# from holdout — not volatility — to avoid a circular import (volatility imports make_regressor
# from this module).
from mmi.ml.holdout import MIN_OBS as _MIN_OBS
from mmi.ml.holdout import split_indices
from mmi.utils.logging import get_logger

log = get_logger("ml.forecast")

# Fixed seed for reproducibility across CV and final fit.
SEED: int = 0


def make_regressor(n_estimators: int, seed: int = SEED) -> RandomForestRegressor:
    # max_depth/min_samples_leaf/max_features control variance at ~1e-4 daily-return scale,
    # where an unconstrained forest over-adapts to noise rather than signal.
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def train_and_backtest(con, symbol: str = "SPY") -> tuple[dict, dict]:
    """Walk-forward backtest + a next-day forecast. Returns (metrics, forecast)."""
    df = con.execute(
        "select date, close, daily_return from marts.fct_asset_daily "
        "where symbol = ? order by date",
        [symbol],
    ).df()

    cols = feature_columns()
    feats = make_features(df)
    # Keep only rows with valid FEATURES (no target yet — see the label-leak note below).
    feat_valid = feats.dropna(subset=cols).reset_index(drop=True)
    n_feat = len(feat_valid)

    # --- carve the locked holdout FIRST, then build the target within each slice ---
    # P1-A (label-leak): the target is ret.shift(-1) (horizon 1), so dev's LAST row would read
    # the FIRST holdout-period return — and that row sits in the final CV test fold, leaking
    # holdout data into the GATED dir_acc/mae.  We carve the feature-valid rows first and
    # rebuild the target separately on dev and on holdout: each slice's last row gets a NaN
    # target (forward window outside the slice) and drops out, exactly like the series end.
    dev_end, hold = split_indices(n_feat, min_dev=_MIN_OBS)
    if hold == 0:
        log.info(
            "direction holdout skipped for %s: %d feature-valid rows too few to carve a "
            "holdout and keep >= %d dev rows — CV runs on the full series",
            symbol,
            n_feat,
            _MIN_OBS,
        )

    def _slice_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild the next-day target WITHIN `frame` and drop rows with a NaN target.

        Building target = ret.shift(-1) here (not on the full series) is what makes the
        dev/holdout label sets disjoint — no dev label reads a holdout-period return.
        """
        f = frame.copy()
        f["target_next_ret"] = f["ret"].shift(-1)
        f = f.dropna(subset=["target_next_ret"])
        return f[cols].to_numpy(), f["target_next_ret"].to_numpy()

    x_dev, y_dev = _slice_xy(feat_valid.iloc[:dev_end])

    n = len(y_dev)  # trainable DEV rows (after the dev-only next-day-target dropna)
    if n < _MIN_OBS:
        raise ValueError(f"not enough observations for {symbol} ({n})")

    # --- walk-forward evaluation (DEV portion only) ---
    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals = [], []
    for train_idx, test_idx in tscv.split(x_dev):
        model = make_regressor(n_estimators=100)
        model.fit(x_dev[train_idx], y_dev[train_idx])
        preds.append(model.predict(x_dev[test_idx]))
        actuals.append(y_dev[test_idx])
    preds_arr = np.concatenate(preds)
    actuals_arr = np.concatenate(actuals)

    mae = float(np.mean(np.abs(preds_arr - actuals_arr)))
    dir_acc = float(np.mean(np.sign(preds_arr) == np.sign(actuals_arr)))
    # Baselines: predict-zero (for MAE) and majority-direction (for accuracy).
    baseline_mae = float(np.mean(np.abs(actuals_arr)))
    baseline_dir = float(max((actuals_arr > 0).mean(), (actuals_arr <= 0).mean()))

    metrics = {
        "symbol": symbol,
        # n_obs is the trainable DEV count the CV actually ran on (== full trainable count when
        # the holdout is skipped).
        "n_obs": int(n),
        "mae": mae,
        "baseline_mae": baseline_mae,
        "dir_acc": dir_acc,
        "baseline_dir_acc": baseline_dir,
    }

    # --- locked holdout: final-fit on ALL of dev, score the untouched holdout slice ---
    # Reported, NOT gated; never used to tune anything.  Skipped (no keys) when hold == 0.
    # The holdout target is built WITHIN the holdout slice (P1-A), so it never reads a dev
    # value and the dev/holdout label sets are disjoint.
    if hold > 0:
        x_hold, y_hold = _slice_xy(feat_valid.iloc[dev_end:])
        if len(y_hold) > 0:
            hold_model = make_regressor(n_estimators=200).fit(x_dev, y_dev)
            hold_preds = hold_model.predict(x_hold)
            holdout_dir_acc = float(np.mean(np.sign(hold_preds) == np.sign(y_hold)))
            # Majority-class baseline on the holdout actuals (matches the CV's baseline_dir).
            holdout_baseline_dir_acc = float(max((y_hold > 0).mean(), (y_hold <= 0).mean()))
            metrics["holdout_dir_acc"] = holdout_dir_acc
            metrics["holdout_baseline_dir_acc"] = holdout_baseline_dir_acc
            metrics["holdout_n_obs"] = float(len(y_hold))
            log.info(
                "direction holdout %s: holdout_dir_acc=%.3f (baseline %.3f) n=%d",
                symbol,
                holdout_dir_acc,
                holdout_baseline_dir_acc,
                len(y_hold),
            )

    # --- final model on all data -> forecast the next day ---
    # The live forecast legitimately uses ALL valid rows (the holdout is an evaluation device).
    full = feat_valid.copy()
    full["target_next_ret"] = full["ret"].shift(-1)
    full_trained = full.dropna(subset=["target_next_ret"])
    final = make_regressor(n_estimators=200).fit(
        full_trained[cols].to_numpy(), full_trained["target_next_ret"].to_numpy()
    )
    next_ret = float(final.predict(feat_valid[cols].iloc[[-1]].to_numpy())[0])

    forecast = {
        "symbol": symbol,
        "as_of": pd.to_datetime(feat_valid["date"].iloc[-1]),
        "predicted_next_return": next_ret,
        "model": "random_forest",
    }
    log.info("backtest %s: dir_acc=%.3f (baseline %.3f)", symbol, dir_acc, baseline_dir)
    return metrics, forecast
