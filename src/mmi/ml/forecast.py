"""Regime-aware, multi-horizon return forecasting with honest walk-forward backtest.

Predicts SPY returns at horizons 1d, 5d, 10d, 20d using a rich feature set (35 features)
and regime-aware Random Forest models (separate models per vol regime: Low/Med/High).

Architecture
------------
- Target: cumulative forward return over ``horizon`` trading days.
- Features: vol_rich set (HAR cascade + kurtosis/skewness + vol-of-vol + cross-asset
  correlations + macro + calendar effects).
- Regime: per-row vol terciles from ``fct_regime`` (Low / Medium / High).
- Model: RandomForestRegressor, per-regime when regime_aware=True.
- Walk-forward: ``TimeSeriesSplit(n_splits=5)`` with embargo for horizons > 1.
- Metrics: direction accuracy, MAE, R², information coefficient (rank correlation),
  plus per-regime breakdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from mmi.ml.features import feature_columns, make_features
from mmi.ml.holdout import MIN_OBS as _MIN_OBS
from mmi.ml.holdout import split_indices
from mmi.utils.logging import get_logger

log = get_logger("ml.forecast")

SEED: int = 0
_REGIME_LABELS = ["Low", "Medium", "High"]
_MIN_REGIME_ROWS = 30  # minimum training samples per regime to train a regime model


def make_regressor(n_estimators: int = 200, seed: int = SEED) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def _build_horizon_target(ret: pd.Series, horizon: int) -> pd.Series:
    """Cumulative forward return over `horizon` trading days.

    At row t: target = sum(ret[t+1], ..., ret[t+horizon]).
    """
    return ret.rolling(horizon, min_periods=horizon).sum().shift(-horizon)


# ---------------------------------------------------------------------------
# Regime helpers
# ---------------------------------------------------------------------------


def _get_regime_labels(con, symbol: str, frame: pd.DataFrame) -> np.ndarray:
    """Get per-row regime labels (0=Low, 1=Medium, 2=High) for the given frame dates."""
    try:
        dates = frame["date"].tolist()
        min_d, max_d = min(dates), max(dates)
        df = con.execute(
            "select date, vol_20d from marts.fct_asset_daily "
            "where symbol = ? and date between ? and ? and vol_20d is not null "
            "order by date",
            [symbol, min_d, max_d],
        ).df()
        if df.empty:
            return np.ones(len(frame), dtype=int)
        df["regime_int"] = pd.qcut(df["vol_20d"].rank(method="first"), 3, labels=[0, 1, 2])
        regime_map = dict(zip(df["date"], df["regime_int"].astype(int), strict=False))
        return np.array([regime_map.get(d, 1) for d in dates])
    except Exception:
        return np.ones(len(frame), dtype=int)


def _train_per_regime(
    x_train: np.ndarray,
    y_train: np.ndarray,
    regime_train: np.ndarray,
    n_estimators: int = 200,
) -> dict[int, RandomForestRegressor]:
    """Train separate RF models per regime. Returns {regime_int: model}."""
    models = {}
    for r in np.unique(regime_train):
        mask = regime_train == r
        if mask.sum() >= _MIN_REGIME_ROWS:
            model = make_regressor(n_estimators=n_estimators)
            model.fit(x_train[mask], y_train[mask])
            models[int(r)] = model
    return models


def _predict_per_regime(
    x_pred: np.ndarray,
    regime_pred: np.ndarray,
    regime_models: dict[int, RandomForestRegressor],
    global_model: RandomForestRegressor,
) -> np.ndarray:
    """Predict using regime-specific models, falling back to global."""
    preds = np.empty(len(regime_pred))
    for i, r in enumerate(regime_pred):
        model = regime_models.get(int(r), global_model)
        preds[i] = model.predict(x_pred[i : i + 1])[0]
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    preds: np.ndarray,
    actuals: np.ndarray,
    regimes: np.ndarray | None = None,
) -> dict:
    """Compute direction accuracy, MAE, R², IC (information coefficient)."""
    mae = float(np.mean(np.abs(preds - actuals)))
    dir_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))
    baseline_mae = float(np.mean(np.abs(actuals)))
    baseline_dir = float(max((actuals > 0).mean(), (actuals <= 0).mean()))

    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0

    # Information coefficient: rank correlation between predictions and actuals
    ic = 0.0
    if len(preds) > 10:
        ic_val, _ = spearmanr(preds, actuals)
        ic = float(ic_val) if not np.isnan(ic_val) else 0.0

    metrics = {
        "dir_acc": dir_acc,
        "baseline_dir_acc": baseline_dir,
        "mae": mae,
        "baseline_mae": baseline_mae,
        "r2": r2,
        "ic": ic,
        "n_obs": len(actuals),
    }

    # Per-regime breakdown
    if regimes is not None:
        for r in [0, 1, 2]:
            mask = regimes == r
            if mask.sum() > 5:
                metrics[f"dir_acc_{_REGIME_LABELS[r].lower()}"] = float(
                    np.mean(np.sign(preds[mask]) == np.sign(actuals[mask]))
                )

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------


def _walk_forward(
    x: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    n_splits: int,
    n_estimators: int,
    regime_aware: bool,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward CV with embargo. Returns (all_preds, all_actuals, all_regimes).

    The embargo drops the last ``horizon`` rows from each training fold to prevent
    target leakage — for h>1, the test fold's targets overlap with training features.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds_list, actuals_list, regime_list = [], [], []

    for train_idx, test_idx in tscv.split(x):
        # Embargo: drop last `horizon` rows from training to prevent leakage
        embargo_end = max(0, len(train_idx) - horizon)
        train_idx_embargoed = train_idx[:embargo_end]

        x_tr = x[train_idx_embargoed]
        y_tr = y[train_idx_embargoed]
        r_tr = regimes[train_idx_embargoed]
        x_te, r_te = x[test_idx], regimes[test_idx]

        if regime_aware and len(np.unique(r_tr)) > 1:
            regime_models = _train_per_regime(x_tr, y_tr, r_tr, n_estimators)
            global_model = make_regressor(n_estimators)
            global_model.fit(x_tr, y_tr)
            pred = _predict_per_regime(x_te, r_te, regime_models, global_model)
        else:
            model = make_regressor(n_estimators)
            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)

        preds_list.append(pred)
        actuals_list.append(y[test_idx])
        regime_list.append(r_te)

    return np.concatenate(preds_list), np.concatenate(actuals_list), np.concatenate(regime_list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_predict(
    con,
    symbol: str = "SPY",
    feature_set: str = "vol_rich",
    horizons: list[int] | None = None,
    n_splits: int = 5,
    n_estimators: int = 200,
    regime_aware: bool = True,
    macro_df: pd.DataFrame | None = None,
    asset_dfs: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict, list[dict]]:
    """Multi-horizon, regime-aware return forecaster.

    Parameters
    ----------
    horizons:
        Forward return horizons in trading days (default [1, 5, 10, 20]).
    regime_aware:
        Train separate RF models per vol regime (Low/Med/High).

    Returns
    -------
    metrics : dict
        Aggregated metrics across horizons.
    forecasts : list[dict]
        Per-horizon forecast dicts for the ml_forecast mart.
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    # Load data
    df = con.execute(
        "select date, open, high, low, close, daily_return "
        "from marts.fct_asset_daily "
        "where symbol = ? order by date",
        [symbol],
    ).df()

    cols = feature_columns(feature_set=feature_set)
    feats = make_features(df, feature_set=feature_set, macro_df=macro_df, asset_dfs=asset_dfs)
    feat_valid = feats.dropna(subset=cols).reset_index(drop=True)
    n_feat = len(feat_valid)

    # Get regime labels for the full feature-valid frame
    if regime_aware:
        full_regimes = _get_regime_labels(con, symbol, feat_valid)
    else:
        full_regimes = np.ones(n_feat, dtype=int)

    log.info(
        "forecast %s: %d feature-valid rows, regimes: %s",
        symbol,
        n_feat,
        {r: int((full_regimes == r).sum()) for r in [0, 1, 2]},
    )

    all_metrics: dict = {"symbol": symbol, "horizons": {}}
    all_forecasts: list[dict] = []

    if n_feat == 0:
        log.warning("no feature-valid rows for %s — returning empty", symbol)
        return all_metrics, all_forecasts

    as_of = pd.to_datetime(feat_valid["date"].iloc[-1])

    for horizon in horizons:
        log.info("  horizon %d days", horizon)

        # Build target within the full frame
        feat_h = feat_valid.copy()
        feat_h["target_h"] = _build_horizon_target(feat_h["ret"], horizon)
        feat_h = feat_h.dropna(subset=["target_h"])

        if len(feat_h) < _MIN_OBS:
            log.warning("  skip horizon %d: only %d trainable rows", horizon, len(feat_h))
            continue

        # Carve holdout
        dev_end, hold = split_indices(len(feat_h), min_dev=_MIN_OBS)

        # Get regimes aligned with the trainable rows
        h_regimes = full_regimes[feat_h.index]

        x_dev = feat_h[cols].iloc[:dev_end].to_numpy()
        y_dev = feat_h["target_h"].iloc[:dev_end].to_numpy()
        r_dev = h_regimes[:dev_end]

        # Walk-forward CV on dev
        preds, actuals, pred_regimes = _walk_forward(
            x_dev, y_dev, r_dev, n_splits, n_estimators, regime_aware, horizon
        )
        h_metrics = _compute_metrics(preds, actuals, pred_regimes)

        # Holdout evaluation
        if hold > 0:
            x_hold = feat_h[cols].iloc[dev_end:].to_numpy()
            y_hold = feat_h["target_h"].iloc[dev_end:].to_numpy()
            r_hold = h_regimes[dev_end:]

            if len(y_hold) > 0:
                if regime_aware and len(np.unique(r_dev)) > 1:
                    r_models = _train_per_regime(x_dev, y_dev, r_dev, n_estimators)
                    g_model = make_regressor(n_estimators)
                    g_model.fit(x_dev, y_dev)
                    hold_preds = _predict_per_regime(x_hold, r_hold, r_models, g_model)
                else:
                    model = make_regressor(n_estimators)
                    model.fit(x_dev, y_dev)
                    hold_preds = model.predict(x_hold)

                hold_metrics = _compute_metrics(hold_preds, y_hold, r_hold)
                h_metrics["holdout_dir_acc"] = hold_metrics["dir_acc"]
                h_metrics["holdout_baseline_dir_acc"] = hold_metrics["baseline_dir_acc"]
                h_metrics["holdout_r2"] = hold_metrics["r2"]
                h_metrics["holdout_ic"] = hold_metrics["ic"]
                h_metrics["holdout_n_obs"] = len(y_hold)

        # Final model on all data → live forecast
        x_full = feat_h[cols].to_numpy()
        y_full = feat_h["target_h"].to_numpy()
        r_full = h_regimes

        if regime_aware and len(np.unique(r_full)) > 1:
            r_models = _train_per_regime(x_full, y_full, r_full, n_estimators)
            g_model = make_regressor(n_estimators)
            g_model.fit(x_full, y_full)
            x_last = feat_h[cols].iloc[[-1]].to_numpy()
            r_last = h_regimes[[-1]]
            predicted = float(_predict_per_regime(x_last, r_last, r_models, g_model)[0])
        else:
            model = make_regressor(n_estimators)
            model.fit(x_full, y_full)
            x_last = feat_h[cols].iloc[[-1]].to_numpy()
            predicted = float(model.predict(x_last)[0])

        # Daily-equivalent for portfolio compatibility
        daily_mu = predicted / horizon

        all_metrics["horizons"][horizon] = h_metrics
        all_forecasts.append(
            {
                "symbol": symbol,
                "as_of": as_of,
                "horizon": horizon,
                "predicted_return": predicted,
                "daily_mu": daily_mu,
                "model": "return_rf_regime" if regime_aware else "return_rf",
                "dir_acc": h_metrics["dir_acc"],
                "r2": h_metrics["r2"],
            }
        )

        log.info(
            "  h=%d: dir_acc=%.3f (baseline %.3f) R²=%.3f IC=%.3f",
            horizon,
            h_metrics["dir_acc"],
            h_metrics["baseline_dir_acc"],
            h_metrics["r2"],
            h_metrics["ic"],
        )

    # Aggregate across horizons
    if all_metrics["horizons"]:
        h1 = all_metrics["horizons"].get(1, {})
        all_metrics["overall_dir_acc"] = h1.get("dir_acc", 0)
        all_metrics["overall_r2"] = h1.get("r2", 0)
        all_metrics["overall_ic"] = h1.get("ic", 0)

    return all_metrics, all_forecasts
