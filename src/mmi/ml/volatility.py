"""HAR realized-volatility model: forward next-week (5 trading-day) forecast for SPY.

Model tag: ``rv_har``

Architecture
------------
- Target: mean Garman-Klass vol over the NEXT 5 trading days (annualised scale not applied;
  we forecast the daily-scale value so it's directly comparable to the persistence baseline).
- Features: ``feature_set='vol'`` from ``mmi.ml.features`` (GK + HAR cascade + macro) using
  only strictly-past data.  The target is ``shift(-5)`` forward from row t so row t's features
  never see the label window.
- Walk-forward ``TimeSeriesSplit(5)``; ``make_regressor()`` from the shared factory.
- Honest baseline: persistence (yesterday's GK vol) and EWMA (RiskMetrics λ=0.94).
- Metrics (long rows, ``model='rv_har'``): ``oos_r2``, ``qlike``, ``baseline_qlike``,
  ``qlike_skill_ratio``, ``n_folds``, ``folds_passed``.

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
from sklearn.model_selection import TimeSeriesSplit

from mmi.ml.features import feature_columns, make_features
from mmi.ml.forecast import make_regressor
from mmi.ml.holdout import split_indices
from mmi.utils.logging import get_logger

log = get_logger("ml.volatility")

# Minimum observations required to attempt training.
_MIN_OBS: int = 60
# Forward horizon in trading days.
_HORIZON: int = 5
# EWMA decay for the persistence/RiskMetrics baseline.
_EWMA_LAMBDA: float = 0.94
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

    h  = predicted variance (pred²)
    sigma² = realised variance (actual²)
    Robust to near-zero: clips to 1e-10.
    """
    h = np.clip(preds**2, 1e-10, None)
    sig2 = np.clip(actuals**2, 1e-10, None)
    ratio = h / sig2
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _make_targets(gk: pd.Series, horizon: int = _HORIZON) -> pd.Series:
    """Forward realized vol = mean of next `horizon` GK values.

    At row t: target = mean(gk[t+1], ..., gk[t+horizon]).
    Rows where gk is shifted forward use only future data that's not yet in any feature.
    """
    return gk.shift(-1).rolling(horizon, min_periods=horizon).mean().shift(-(horizon - 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_backtest_vol(con, symbol: str = "SPY") -> tuple[dict, dict | None]:
    """Walk-forward HAR vol backtest + next-week forecast.

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

    feats = make_features(df, feature_set="vol")
    vol_cols = feature_columns(feature_set="vol")

    # Build the forward-realized-vol target: mean(gk[t+1..t+5])
    feats["target_rv"] = _make_targets(feats["gk_vol"], horizon=_HORIZON)

    # Drop rows where either features or target are NaN
    valid = feats.dropna(subset=vol_cols + ["target_rv"])

    n = len(valid)
    if n < _MIN_OBS:
        log.warning(
            "skip vol model for %s: only %d valid rows (need %d)",
            symbol,
            n,
            _MIN_OBS,
        )
        return {}, None

    x = valid[vol_cols].to_numpy()
    y = valid["target_rv"].to_numpy()  # forward realized vol (positive daily scale)
    # GK vol aligned to the x/y evaluation rows; the EWMA (RiskMetrics λ=0.94) baseline
    # is derived from this per fold, never from the full series (see below).
    gk_valid = valid["gk_vol"].to_numpy()

    # --- carve the locked holdout (honest extra OOS readout; reported, not gated) ---
    # The holdout is the LAST `hold` time-ordered rows; everything before is DEV.  The
    # walk-forward CV (and therefore the skill gate) runs on DEV ONLY — the holdout is never
    # in a training/test fold.  If carving would leave < _MIN_OBS dev rows we skip it.
    dev_end, hold = split_indices(n, min_dev=_MIN_OBS)
    if hold == 0:
        log.info(
            "vol holdout skipped for %s: %d valid rows too few to carve a holdout "
            "and keep >= %d dev rows — CV runs on the full series",
            symbol,
            n,
            _MIN_OBS,
        )
    x_dev, y_dev, gk_dev = x[:dev_end], y[:dev_end], gk_valid[:dev_end]

    # --- walk-forward evaluation (DEV portion only) ---
    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals, base_preds = [], [], []
    for train_idx, test_idx in tscv.split(x_dev):
        model = make_regressor(n_estimators=100)
        model.fit(x_dev[train_idx], y_dev[train_idx])
        preds.append(model.predict(x_dev[test_idx]))
        actuals.append(y_dev[test_idx])
        # EWMA baseline computed walk-forward: only gk vol up to each fold's last test
        # point feeds the recursion — no look-ahead (honest-OOS contract).
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
    n_folds = 5
    folds_passed = 0
    for p, a, b in zip(preds, actuals, base_preds, strict=False):
        if _qlike(a, p) < _qlike(a, b):
            folds_passed += 1

    metrics = {
        "symbol": symbol,
        # n_obs is the DEV count the CV actually ran on (== n when the holdout is skipped),
        # so the gate's n_obs check reflects the data behind oos_r2/qlike — not rows held out.
        "n_obs": dev_end,
        "oos_r2": oos_r2,
        "qlike": qlike,
        "baseline_qlike": baseline_qlike,
        "qlike_skill_ratio": qlike_skill_ratio,
        "n_folds": n_folds,
        "folds_passed": folds_passed,
    }

    # --- locked holdout: final-fit on ALL of dev, score the untouched tail ---
    # Reported, NOT gated — these holdout_* rows never enter skill_verdict() and are NEVER
    # used to tune anything.  When the holdout was skipped (hold == 0) we emit no rows.
    if hold > 0:
        x_hold, y_hold = x[dev_end:], y[dev_end:]
        hold_model = make_regressor(n_estimators=200)
        hold_model.fit(x_dev, y_dev)
        hold_preds = hold_model.predict(x_hold)

        ss_res_h = float(np.sum((y_hold - hold_preds) ** 2))
        ss_tot_h = float(np.sum((y_hold - y_hold.mean()) ** 2))
        holdout_oos_r2 = 1.0 - ss_res_h / ss_tot_h if ss_tot_h > 1e-20 else 0.0

        # Same persistence/EWMA baseline as the CV, evaluated on the holdout tail.  The EWMA
        # recursion is seeded from all rows up to and including each holdout point (causal,
        # no look-ahead), exactly as the walk-forward baseline does per fold.
        hold_idx = np.arange(dev_end, n)
        holdout_base = _walk_forward_ewma_baseline(gk_valid, hold_idx)
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
        metrics["holdout_n_obs"] = hold

        log.info(
            "vol holdout %s: holdout_oos_r2=%.3f holdout_qlike_ratio=%.3f n=%d",
            symbol,
            holdout_oos_r2,
            holdout_qlike_skill_ratio if not np.isnan(holdout_qlike_skill_ratio) else -1,
            hold,
        )

    # --- final model on all data -> forecast next week ---
    # The live forecast legitimately uses ALL valid rows (the holdout is an evaluation device,
    # not a data quarantine for the production prediction).
    final = make_regressor(n_estimators=200)
    final.fit(x, y)
    next_vol = float(final.predict(x[[-1]])[0])

    forecast = {
        "symbol": symbol,
        "as_of": pd.to_datetime(valid["date"].iloc[-1]),
        "predicted_next_return": next_vol,  # field name matches ml_forecast schema
        "model": MODEL_TAG,
    }

    log.info(
        "vol backtest %s: oos_r2=%.3f qlike_ratio=%.3f folds_passed=%d/%d",
        symbol,
        oos_r2,
        qlike_skill_ratio if not np.isnan(qlike_skill_ratio) else -1,
        folds_passed,
        n_folds,
    )
    return metrics, forecast
