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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from mmi.ml.features import feature_columns, make_features
from mmi.ml.holdout import split_indices
from mmi.utils.logging import get_logger

log = get_logger("ml.forecast")

# Fixed seed for reproducibility across CV and final fit.
SEED: int = 0
# Minimum observations required to attempt training (mirrors volatility._MIN_OBS).
_MIN_OBS: int = 60


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

    feats = make_features(df).dropna(subset=feature_columns() + ["target_next_ret"])
    cols = feature_columns()
    x = feats[cols].to_numpy()
    y = feats["target_next_ret"].to_numpy()

    n = len(y)
    if n < _MIN_OBS:
        raise ValueError(f"not enough observations for {symbol} ({n})")

    # --- carve the locked holdout (honest extra OOS readout; reported, not gated) ---
    # The holdout is the LAST `hold` time-ordered rows; everything before is DEV.  The
    # walk-forward CV (which drives the reported dir_acc/mae) runs on DEV ONLY — the holdout
    # is never in a training/test fold.  If carving would leave < _MIN_OBS dev rows we skip it.
    dev_end, hold = split_indices(n, min_dev=_MIN_OBS)
    if hold == 0:
        log.info(
            "direction holdout skipped for %s: %d valid rows too few to carve a holdout "
            "and keep >= %d dev rows — CV runs on the full series",
            symbol,
            n,
            _MIN_OBS,
        )
    x_dev, y_dev = x[:dev_end], y[:dev_end]

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
        # n_obs is the DEV count the CV actually ran on (== n when the holdout is skipped).
        "n_obs": int(dev_end),
        "mae": mae,
        "baseline_mae": baseline_mae,
        "dir_acc": dir_acc,
        "baseline_dir_acc": baseline_dir,
    }

    # --- locked holdout: final-fit on ALL of dev, score the untouched tail ---
    # Reported, NOT gated; never used to tune anything.  Skipped (no keys) when hold == 0.
    if hold > 0:
        x_hold, y_hold = x[dev_end:], y[dev_end:]
        hold_model = make_regressor(n_estimators=200).fit(x_dev, y_dev)
        hold_preds = hold_model.predict(x_hold)
        holdout_dir_acc = float(np.mean(np.sign(hold_preds) == np.sign(y_hold)))
        # Majority-class baseline on the holdout actuals (matches the CV's baseline_dir rule).
        holdout_baseline_dir_acc = float(max((y_hold > 0).mean(), (y_hold <= 0).mean()))
        metrics["holdout_dir_acc"] = holdout_dir_acc
        metrics["holdout_baseline_dir_acc"] = holdout_baseline_dir_acc
        metrics["holdout_n_obs"] = float(hold)
        log.info(
            "direction holdout %s: holdout_dir_acc=%.3f (baseline %.3f) n=%d",
            symbol,
            holdout_dir_acc,
            holdout_baseline_dir_acc,
            hold,
        )

    # --- final model on all data -> forecast the next day ---
    # The live forecast legitimately uses ALL valid rows (the holdout is an evaluation device).
    final = make_regressor(n_estimators=200).fit(x, y)
    next_ret = float(final.predict(feats[cols].iloc[[-1]].to_numpy())[0])

    forecast = {
        "symbol": symbol,
        "as_of": pd.to_datetime(feats["date"].iloc[-1]),
        "predicted_next_return": next_ret,
        "model": "random_forest",
    }
    log.info("backtest %s: dir_acc=%.3f (baseline %.3f)", symbol, dir_acc, baseline_dir)
    return metrics, forecast
