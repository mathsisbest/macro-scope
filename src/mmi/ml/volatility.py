"""HAR realized-volatility model: forward next-week (5 trading-day) forecast for SPY.

Model tag: ``rv_har``

Architecture
------------
- Target: mean Garman-Klass vol over the NEXT 5 trading days (annualised scale not applied;
  we forecast the daily-scale value so it's directly comparable to the persistence baseline).
- Estimator: a classic **log-space OLS HAR** (Corsi 2009) on the daily/weekly/monthly realised-
  vol cascade (``_HAR_COLS``), with Duan smearing for the log→level retransformation bias (see
  :func:`_fit_predict_har`).  This replaces the regularised random forest the model previously
  borrowed from the direction task — a forest cannot extrapolate past its training vol range, so
  calm-train/crisis-test folds structurally under-predicted (negative OOS R²); a linear HAR
  extrapolates, which is the entire point of the HAR result.  OLS has no hyperparameters to tune.
- Target is built ``shift(-5)`` forward from row t so row t's features never see the label window.
- Walk-forward ``TimeSeriesSplit(5)``.
- Honest baseline: persistence (yesterday's GK vol) and EWMA (RiskMetrics λ=0.94).
- Metrics (long rows, ``model='rv_har'``): ``oos_r2``, ``qlike``, ``baseline_qlike``,
  ``qlike_skill_ratio``, ``n_folds``, ``folds_passed``.  QLIKE floors realised vol at
  ``_VOL_FLOOR`` (identically for model and baseline) so a near-flat day can't make the loss
  explode — the fix for the under-specified-baseline red flag.

Small-sample safety
-------------------
If the DB has fewer than ``_MIN_OBS`` rows with valid features + target, the function logs a
warning and returns ``({}, None)`` — **no crash** — mirroring the ValueError-caught pattern
already in ``pipeline.py``.

Locked holdout (honest extra OOS readout)
-----------------------------------------
After feature construction we carve off the LAST ``holdout_size`` time-ordered rows as a
locked holdout (see :mod:`mmi.ml.holdout`).  The holdout is **never** passed to the
walk-forward CV or any training fold — the gate metrics (``oos_r2``, ``qlike_skill_ratio``,
``folds_passed`` …) are computed on the DEV portion exactly as before.  We then final-fit one
model on ALL of dev, predict the untouched holdout, and emit ``holdout_*`` metric rows
(``holdout_oos_r2``, ``holdout_qlike``, ``holdout_qlike_skill_ratio``, ``holdout_n_obs``).
The holdout is **reported, not gated** — :func:`mmi.ml.skill_gate.skill_verdict` is unchanged
and never sees these rows.  It is an honest extra out-of-sample readout and is NEVER used to
tune the model, the features, or the gate thresholds.

If carving the holdout would leave fewer than ``_MIN_OBS`` dev rows, the holdout is SKIPPED:
no ``holdout_*`` rows are emitted, the CV runs on the full valid series as before, and the
skip is logged at INFO.  This keeps the small CI/sample data working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

from mmi.ml.features import feature_columns, har_feature_names, make_features
from mmi.ml.holdout import MIN_OBS as _MIN_OBS
from mmi.ml.holdout import split_indices
from mmi.utils.logging import get_logger

log = get_logger("ml.volatility")

# Minimum observations required to attempt training.  Canonical value lives in mmi.ml.holdout
# (shared with the direction model); re-exported here so existing `from volatility import
# _MIN_OBS` call sites keep working.
# Forward horizon in trading days.
_HORIZON: int = 5
# EWMA decay for the persistence/RiskMetrics baseline.
_EWMA_LAMBDA: float = 0.94
# Realised-vol floor (daily scale), used (a) before the log transform in the HAR estimator and
# (b) inside QLIKE.  ~0.2%/day ≈ 3.2% annualised — comfortably below any sustained market vol, so
# it only clips genuinely degenerate near-flat windows where Garman-Klass collapses toward 0.
# QLIKE divides by realised variance, so a near-zero proxy makes the ratio explode; flooring it
# (identically for model AND baseline) bounds the loss so a handful of flat days can't dominate.
# Economically motivated and FIXED — never tuned to make a run clear the gate.
_VOL_FLOOR: float = 0.002
# HAR cascade columns (Corsi 2009): trailing daily / weekly / monthly realised-vol averages.
# Derived from features.har_feature_names() (the single source of truth, built from
# features._VOL_HAR_WINDOWS) so this list can never silently drift from the columns
# make_features actually produces.
_HAR_COLS: list[str] = har_feature_names()
# Model tag — used as the 'model' column value in marts.model_metrics + marts.ml_forecast.
MODEL_TAG: str = "rv_har"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ewma_vol(gk: pd.Series, lam: float = _EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics EWMA of GK vol.  ewma[t] = lam*ewma[t-1] + (1-lam)*gk[t].

    Uses pandas ewm(alpha=1-lam) which initialises from the first observation.
    """
    return gk.ewm(alpha=1.0 - lam, adjust=False).mean()


def _walk_forward_ewma_baseline(
    gk: np.ndarray, test_idx: np.ndarray, lam: float = _EWMA_LAMBDA
) -> np.ndarray:
    """EWMA baseline predictions for one walk-forward fold — honest-OOS, no look-ahead.

    The baseline forecast at a test point is the RiskMetrics EWMA *level* known at that
    point.  We recompute the EWMA over ``gk`` truncated at the fold's LAST test row, so
    rows beyond the test window never enter the recursion at all.  Because the EWMA
    (``adjust=False``) is causal, the value at every test point depends only on ``gk`` up
    to and including that point — there is no forward dependence on data the fold could
    not have seen.

    This replaces the earlier full-series EWMA (computed once, then indexed per fold),
    whose leak-free-ness silently relied on the reader knowing ``ewm`` is causal.
    Computing it per fold over a truncated series makes the walk-forward contract
    structural — and the ``test_ewma_baseline_no_forward_dependence`` regression guards it.
    """
    upto = int(test_idx.max()) + 1
    ewma = _ewma_vol(pd.Series(gk[:upto]), lam=lam)
    return ewma.to_numpy()[test_idx]


def _qlike(actuals: np.ndarray, preds: np.ndarray) -> float:
    """QLIKE loss: mean(h/sigma² - log(h/sigma²) - 1).

    h  = predicted variance (floored-pred²)
    sigma² = realised variance (floored-actual²)

    Both the prediction and the realised proxy are floored at ``_VOL_FLOOR`` (a vol-level floor)
    BEFORE squaring.  QLIKE divides predicted by realised variance, so without a floor a
    near-zero realised proxy (a flat Garman-Klass window) sends the ratio to ~1e6+ and a single
    day dominates the mean — which is exactly what made ``baseline_qlike`` nonsensically large and
    ``qlike_skill_ratio`` a free pass.  The floor is applied IDENTICALLY to model and baseline so
    the comparison stays fair, and it bounds both tails of the ratio (a near-zero prediction blows
    up ``-log(ratio)`` just as a near-zero realised value blows up ``ratio``).
    """
    h = np.clip(preds, _VOL_FLOOR, None) ** 2
    sig2 = np.clip(actuals, _VOL_FLOOR, None) ** 2
    ratio = h / sig2
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _make_targets(gk: pd.Series, horizon: int = _HORIZON) -> pd.Series:
    """Forward realized vol = mean of next `horizon` GK values.

    At row t: target = mean(gk[t+1], ..., gk[t+horizon]).
    Rows where gk is shifted forward use only future data that's not yet in any feature.
    """
    return gk.shift(-1).rolling(horizon, min_periods=horizon).mean().shift(-(horizon - 1))


def _fit_predict_har(
    x_train_log: np.ndarray, y_train_vol: np.ndarray, x_pred_log: np.ndarray
) -> np.ndarray:
    """Fit a log-space OLS HAR and return vol-LEVEL predictions for ``x_pred_log``.

    Classic HAR (Corsi 2009): forward realised vol is approximately linear in the daily / weekly
    / monthly cascade of past realised vol.  We fit ordinary least squares in LOG space (realised
    vol is right-skewed; logs keep the relationship linear and predictions strictly positive),
    then exponentiate back to the vol level with Duan's (1983) smearing estimator —
    ``mean(exp(train residuals))`` — to correct the log→level retransformation bias.

    Inputs are the log HAR-cascade design (already floored + logged by :func:`_slice_xy`) and the
    vol-LEVEL training target.  OLS has NO hyperparameters, and the smearing factor is derived
    from the training residuals — so nothing in here can be tuned to chase the gate.  A forest
    (the prior estimator) cannot extrapolate past its training vol range, which is why
    calm-train/crisis-test folds collapsed to negative R²; the linear HAR extrapolates.
    """
    ylog = np.log(np.clip(y_train_vol, _VOL_FLOOR, None))
    model = LinearRegression().fit(x_train_log, ylog)
    smear = float(np.mean(np.exp(ylog - model.predict(x_train_log))))
    return np.exp(model.predict(x_pred_log)) * smear


def _fit_predict_ridge(
    x_train_log: np.ndarray,
    y_train_vol: np.ndarray,
    x_pred_log: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Ridge regression in log space + Duan smearing (same as HAR but regularised)."""
    ylog = np.log(np.clip(y_train_vol, _VOL_FLOOR, None))
    model = Ridge(alpha=alpha).fit(x_train_log, ylog)
    smear = float(np.mean(np.exp(ylog - model.predict(x_train_log))))
    return np.exp(model.predict(x_pred_log)) * smear


def _fit_predict_lasso(
    x_train_log: np.ndarray,
    y_train_vol: np.ndarray,
    x_pred_log: np.ndarray,
    alpha: float = 0.01,
) -> np.ndarray:
    """Lasso regression in log space + Duan smearing."""
    ylog = np.log(np.clip(y_train_vol, _VOL_FLOOR, None))
    model = Lasso(alpha=alpha, max_iter=5000).fit(x_train_log, ylog)
    smear = float(np.mean(np.exp(ylog - model.predict(x_train_log))))
    return np.exp(model.predict(x_pred_log)) * smear


def _fit_predict_gb(
    x_train: np.ndarray,
    y_train_vol: np.ndarray,
    x_pred: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 3,
) -> np.ndarray:
    """Gradient Boosting in level space (no log transform needed — trees handle skew)."""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=20,
        random_state=0,
    ).fit(x_train, y_train_vol)
    return model.predict(x_pred)


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
            return np.ones(len(frame), dtype=int)  # default to Medium
        # Label regimes via terciles
        df["regime"] = pd.qcut(df["vol_20d"].rank(method="first"), 3, labels=[0, 1, 2])
        # Merge onto frame dates
        regime_map = dict(zip(df["date"], df["regime"].astype(int), strict=False))
        return np.array([regime_map.get(d, 1) for d in dates])
    except Exception:
        return np.ones(len(frame), dtype=int)  # default to Medium


def _fit_predict_har_regime(
    x_train: np.ndarray,
    y_train_vol: np.ndarray,
    x_pred: np.ndarray,
    regime_train: np.ndarray,
    regime_pred: np.ndarray,
    min_regime_rows: int = 30,
) -> np.ndarray:
    """Regime-aware HAR: train separate models per regime, predict with the regime-specific model.

    Falls back to the global model when a regime has fewer than ``min_regime_rows`` training
    samples.
    """
    ylog = np.log(np.clip(y_train_vol, _VOL_FLOOR, None))
    unique_regimes = np.unique(regime_train)

    # Pre-train a global model as fallback
    global_model = LinearRegression().fit(x_train, ylog)
    global_smear = float(np.mean(np.exp(ylog - global_model.predict(x_train))))

    # Train per-regime models
    regime_models = {}
    regime_smeans = {}
    for r in unique_regimes:
        mask = regime_train == r
        if mask.sum() >= min_regime_rows:
            x_r = x_train[mask]
            y_r = ylog[mask]
            model = LinearRegression().fit(x_r, y_r)
            smear = float(np.mean(np.exp(y_r - model.predict(x_r))))
            regime_models[r] = model
            regime_smeans[r] = smear

    # Predict: use regime-specific model if available, else global
    preds = np.empty(len(regime_pred))
    for i, r in enumerate(regime_pred):
        if r in regime_models:
            p = np.exp(regime_models[r].predict(x_pred[i : i + 1]))[0] * regime_smeans[r]
        else:
            p = np.exp(global_model.predict(x_pred[i : i + 1]))[0] * global_smear
        preds[i] = float(p)
    return preds


# Model registry: maps model_name -> (fit_function, needs_log_transform)
# rv_har_regime is handled specially in the walk-forward loop (not a simple fit/predict fn).
_MODEL_REGISTRY: dict[str, tuple] = {
    "rv_har": (_fit_predict_har, True),
    "rv_ridge": (_fit_predict_ridge, True),
    "rv_lasso": (_fit_predict_lasso, True),
    "rv_gb": (_fit_predict_gb, False),
    "rv_har_regime": (_fit_predict_har, True),  # regime logic is in the walk-forward loop
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_backtest_vol(
    con,
    symbol: str = "SPY",
    model_name: str = "rv_har",
    feature_set: str = "vol",
    horizon: int = _HORIZON,
    n_splits: int = 5,
    min_dev: int = _MIN_OBS,
    model_params: dict | None = None,
    macro_df: pd.DataFrame | None = None,
    asset_dfs: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict, dict | None]:
    """Walk-forward vol backtest + next-week forecast.

    Parameters
    ----------
    model_name:
        Model to use: ``'rv_har'``, ``'rv_ridge'``, ``'rv_lasso'``, ``'rv_gb'``.
    feature_set:
        ``'vol'`` (default HAR features) or ``'vol_macro'`` (adds macro/cross-asset).
    horizon:
        Forward target horizon in trading days (default 5 = next-week).
    n_splits:
        Number of walk-forward CV folds.
    min_dev:
        Minimum trainable dev rows required.
    model_params:
        Extra kwargs passed to the fit function (e.g. ``{'alpha': 0.1}``).
    macro_df:
        Macro dataframe for ``vol_macro`` feature set.
    asset_dfs:
        Per-symbol daily data for cross-asset vol features.

    Returns
    -------
    metrics : dict
        Flat metric dict (keys are metric names, values are floats).
        Empty dict on small-sample skip.
    forecast : dict | None
        ``{symbol, as_of, predicted_next_return, model}`` — matches ``ml_forecast`` schema.
        ``None`` on small-sample skip.
    """
    df = con.execute(
        "select date, open, high, low, close, daily_return "
        "from marts.fct_asset_daily "
        "where symbol = ? order by date",
        [symbol],
    ).df()

    feats = make_features(df, feature_set=feature_set, macro_df=macro_df, asset_dfs=asset_dfs)
    vol_cols = feature_columns(feature_set=feature_set)

    # Resolve model
    fit_fn, needs_log = _MODEL_REGISTRY.get(model_name, (_fit_predict_har, True))
    extra_params = model_params or {}

    # Keep only rows with valid FEATURES (no target yet — see the label-leak note below).
    feat_valid = feats.dropna(subset=vol_cols).reset_index(drop=True)
    n_feat = len(feat_valid)

    # --- carve the locked holdout FIRST, then build the target within each slice ---
    # P1-A (label-leak): the forward target mean(gk[t+1..t+horizon]) must be built on the DEV
    # slice ALONE, otherwise dev's last `horizon` rows would read gk values from the holdout
    # period — and those rows sit in the final CV test fold, leaking holdout data into the
    # GATED metrics (oos_r2/qlike).  By splitting the feature-valid rows first and building the
    # target separately on dev and on holdout, each slice's last `horizon` rows get an
    # incomplete forward window -> NaN -> dropped (exactly like the natural end of a series).
    # The carve is purely feature-based and time-ordered; the skip guard is unchanged.
    dev_end, hold = split_indices(n_feat, min_dev=min_dev)
    if hold == 0:
        log.info(
            "vol holdout skipped for %s: %d feature-valid rows too few to carve a holdout "
            "and keep >= %d dev rows — CV runs on the full series",
            symbol,
            n_feat,
            min_dev,
        )

    def _slice_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the forward target WITHIN `frame` and drop rows with a NaN target.

        Returns ``(x, y, gk)`` where ``x`` is the design matrix (logged if needed),
        ``y`` the vol-LEVEL forward target, ``gk`` the daily GK vol (for the EWMA baseline).
        """
        f = frame.copy()
        f["target_rv"] = _make_targets(f["gk_vol"], horizon=horizon)
        f = f.dropna(subset=["target_rv"])
        x_raw = f[vol_cols].to_numpy()
        x = np.log(np.clip(x_raw, _VOL_FLOOR, None)) if needs_log else x_raw
        return (
            x,
            f["target_rv"].to_numpy(),
            f["gk_vol"].to_numpy(),
        )

    dev_frame = feat_valid.iloc[:dev_end]
    x_dev, y_dev, gk_dev = _slice_xy(dev_frame)

    n = len(y_dev)  # trainable DEV rows (after the dev-only forward-target dropna)
    if n < min_dev:
        log.warning(
            "skip vol model for %s: only %d trainable dev rows (need %d)",
            symbol,
            n,
            min_dev,
        )
        return {}, None

    # --- walk-forward evaluation (DEV portion only) ---
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds, actuals, base_preds = [], [], []

    # Regime labels for regime-aware model — built from the dev FRAME (before target dropna),
    # then trimmed to match the actual training rows (after target dropna removes the last
    # `horizon` rows per slice).
    regime_labels = None
    if model_name == "rv_har_regime":
        dev_regime_full = _get_regime_labels(con, symbol, dev_frame)
        # Trim to match the trainable rows (same indices that _slice_xy keeps)
        target_rv = _make_targets(dev_frame["gk_vol"], horizon=horizon)
        valid_mask = target_rv.notna().to_numpy()
        regime_labels = dev_regime_full[valid_mask]

    for train_idx, test_idx in tscv.split(x_dev):
        if model_name == "rv_har_regime" and regime_labels is not None:
            preds.append(
                _fit_predict_har_regime(
                    x_dev[train_idx],
                    y_dev[train_idx],
                    x_dev[test_idx],
                    regime_labels[train_idx],
                    regime_labels[test_idx],
                )
            )
        else:
            preds.append(
                fit_fn(x_dev[train_idx], y_dev[train_idx], x_dev[test_idx], **extra_params)
            )
        actuals.append(y_dev[test_idx])
        base_preds.append(_walk_forward_ewma_baseline(gk_dev, test_idx))

    preds_arr = np.concatenate(preds)
    actuals_arr = np.concatenate(actuals)
    base_arr = np.concatenate(base_preds)

    # OOS R² (Mincer-Zarnowitz style)
    ss_res = float(np.sum((actuals_arr - preds_arr) ** 2))
    ss_tot = float(np.sum((actuals_arr - actuals_arr.mean()) ** 2))
    oos_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0

    # QLIKE
    qlike = _qlike(actuals_arr, preds_arr)
    baseline_qlike = _qlike(actuals_arr, base_arr)
    qlike_skill_ratio = qlike / baseline_qlike if baseline_qlike > 1e-20 else float("nan")

    # folds_passed: folds where model QLIKE < baseline QLIKE
    n_folds = n_splits
    folds_passed = 0
    for p, a, b in zip(preds, actuals, base_preds, strict=False):
        if _qlike(a, p) < _qlike(a, b):
            folds_passed += 1

    metrics = {
        "symbol": symbol,
        # n_obs is the trainable DEV count the CV actually ran on (== full trainable count when
        # the holdout is skipped), so the gate's n_obs check reflects the data behind
        # oos_r2/qlike — not rows held out, and not rows lost to the forward-target window.
        "n_obs": n,
        "oos_r2": oos_r2,
        "qlike": qlike,
        "baseline_qlike": baseline_qlike,
        "qlike_skill_ratio": qlike_skill_ratio,
        "n_folds": n_folds,
        "folds_passed": folds_passed,
    }

    # --- locked holdout: final-fit on ALL of dev, score the untouched holdout slice ---
    # Reported, NOT gated — these holdout_* rows never enter skill_verdict() and are NEVER
    # used to tune anything.  When the holdout was skipped (hold == 0) we emit no rows.
    # The holdout target is built WITHIN the holdout slice (P1-A), so it never reads a dev
    # value and the dev/holdout label sets are disjoint.
    if hold > 0:
        hold_frame = feat_valid.iloc[dev_end:]
        x_hold, y_hold, gk_hold = _slice_xy(hold_frame)
        if len(y_hold) > 0:
            if model_name == "rv_har_regime":
                hold_regime_full = _get_regime_labels(con, symbol, hold_frame)
                hold_target = _make_targets(hold_frame["gk_vol"], horizon=horizon)
                hold_valid = hold_target.notna().to_numpy()
                hold_regime = hold_regime_full[hold_valid]
                hold_preds = _fit_predict_har_regime(
                    x_dev,
                    y_dev,
                    x_hold,
                    regime_labels,
                    hold_regime,
                )
            else:
                hold_preds = fit_fn(x_dev, y_dev, x_hold, **extra_params)

            ss_res_h = float(np.sum((y_hold - hold_preds) ** 2))
            ss_tot_h = float(np.sum((y_hold - y_hold.mean()) ** 2))
            holdout_oos_r2 = 1.0 - ss_res_h / ss_tot_h if ss_tot_h > 1e-20 else 0.0

            # Same persistence/EWMA baseline as the CV, WARM-started over the full dev history
            # then continued into the holdout slice — seeded exactly like the CV folds (which
            # recurse over gk_dev), NOT cold-started from the holdout's first row.  A cold start
            # would re-seed the λ=0.94 (~16-day memory) recursion from gk_hold[0] with no prior
            # history, biasing roughly the first ~20–30 holdout predictions and making the
            # reported holdout baseline (and therefore holdout_qlike_skill_ratio) unfairly
            # favourable to the model.  EWMA (adjust=False) is strictly causal, so each holdout
            # prediction still depends only on gk up to and including that point — concatenating
            # gk_dev ahead of gk_hold adds prior history without any look-ahead into later
            # holdout values.
            gk_warm = np.concatenate([gk_dev, gk_hold])
            holdout_base = _walk_forward_ewma_baseline(
                gk_warm, np.arange(len(gk_dev), len(gk_warm))
            )
            holdout_qlike = _qlike(y_hold, hold_preds)
            holdout_baseline_qlike = _qlike(y_hold, holdout_base)
            holdout_qlike_skill_ratio = (
                holdout_qlike / holdout_baseline_qlike
                if holdout_baseline_qlike > 1e-20
                else float("nan")
            )

            metrics["holdout_oos_r2"] = holdout_oos_r2
            metrics["holdout_qlike"] = holdout_qlike
            metrics["holdout_qlike_skill_ratio"] = holdout_qlike_skill_ratio
            metrics["holdout_n_obs"] = len(y_hold)

            log.info(
                "vol holdout %s: holdout_oos_r2=%.3f holdout_qlike_ratio=%.3f n=%d",
                symbol,
                holdout_oos_r2,
                holdout_qlike_skill_ratio if not np.isnan(holdout_qlike_skill_ratio) else -1,
                len(y_hold),
            )

    # --- final model on all data -> forecast next week ---
    full = feat_valid.copy()
    full["target_rv"] = _make_targets(full["gk_vol"], horizon=horizon)
    full_trained = full.dropna(subset=["target_rv"])
    x_full_raw = full_trained[vol_cols].to_numpy()
    x_full = np.log(np.clip(x_full_raw, _VOL_FLOOR, None)) if needs_log else x_full_raw
    y_full = full_trained["target_rv"].to_numpy()
    x_last_raw = full[vol_cols].iloc[[-1]].to_numpy()
    x_last = np.log(np.clip(x_last_raw, _VOL_FLOOR, None)) if needs_log else x_last_raw
    if model_name == "rv_har_regime":
        # full_trained already has the target built and NaN rows dropped.
        # Regime labels must align with full_trained's row count.
        full_regime = _get_regime_labels(con, symbol, full_trained)
        last_regime = _get_regime_labels(con, symbol, full.iloc[[-1]])
        next_vol = float(
            _fit_predict_har_regime(
                x_full,
                y_full,
                x_last,
                full_regime,
                last_regime,
            )[0]
        )
    else:
        next_vol = float(fit_fn(x_full, y_full, x_last, **extra_params)[0])

    forecast = {
        "symbol": symbol,
        "as_of": pd.to_datetime(feat_valid["date"].iloc[-1]),
        "predicted_next_return": next_vol,
        "model": model_name,
    }

    log.info(
        "vol backtest %s (%s): oos_r2=%.3f qlike_ratio=%.3f folds_passed=%d/%d",
        symbol,
        model_name,
        oos_r2,
        qlike_skill_ratio if not np.isnan(qlike_skill_ratio) else -1,
        folds_passed,
        n_folds,
    )
    return metrics, forecast
