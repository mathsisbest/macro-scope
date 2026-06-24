"""Next-day return forecasting with an honest walk-forward backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from mmi.ml.features import feature_columns, make_features
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

    feats = make_features(df).dropna(subset=feature_columns() + ["target_next_ret"])
    cols = feature_columns()
    x = feats[cols].to_numpy()
    y = feats["target_next_ret"].to_numpy()

    if len(y) < 60:
        raise ValueError(f"not enough observations for {symbol} ({len(y)})")

    # --- walk-forward evaluation (no shuffling; respects time order) ---
    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals = [], []
    for train_idx, test_idx in tscv.split(x):
        model = make_regressor(n_estimators=100)
        model.fit(x[train_idx], y[train_idx])
        preds.append(model.predict(x[test_idx]))
        actuals.append(y[test_idx])
    preds_arr = np.concatenate(preds)
    actuals_arr = np.concatenate(actuals)

    mae = float(np.mean(np.abs(preds_arr - actuals_arr)))
    dir_acc = float(np.mean(np.sign(preds_arr) == np.sign(actuals_arr)))
    # Baselines: predict-zero (for MAE) and majority-direction (for accuracy).
    baseline_mae = float(np.mean(np.abs(actuals_arr)))
    baseline_dir = float(max((actuals_arr > 0).mean(), (actuals_arr <= 0).mean()))

    # --- final model on all data -> forecast the next day ---
    final = make_regressor(n_estimators=200).fit(x, y)
    next_ret = float(final.predict(feats[cols].iloc[[-1]].to_numpy())[0])

    metrics = {
        "symbol": symbol,
        "n_obs": int(len(y)),
        "mae": mae,
        "baseline_mae": baseline_mae,
        "dir_acc": dir_acc,
        "baseline_dir_acc": baseline_dir,
    }
    forecast = {
        "symbol": symbol,
        "as_of": pd.to_datetime(feats["date"].iloc[-1]),
        "predicted_next_return": next_ret,
        "model": "random_forest",
    }
    log.info("backtest %s: dir_acc=%.3f (baseline %.3f)", symbol, dir_acc, baseline_dir)
    return metrics, forecast
