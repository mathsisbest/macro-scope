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
    dev_end, hold = split_indices(n_feat, min_dev=_MIN_OBS)
    if hold == 0:
        log.info(
            "vol holdout skipped for %s: %d feature-valid rows too few to carve a holdout "
            "and keep >= %d dev rows — CV runs on the full series",
            symbol,
            n_feat,
            _MIN_OBS,
        )

    def _slice_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the forward target WITHIN `frame` and drop rows with a NaN target.

        Returns (x, y, gk) for the rows of `frame` whose forward window is fully inside
        `frame`.  Building the target here (not on the full series) is what makes the
        dev/holdout label sets disjoint — no dev label depends on a holdout-period value.
        """
        f = frame.copy()
        f["target_rv"] = _make_targets(f["gk_vol"], horizon=_HORIZON)
        f = f.dropna(subset=["target_rv"])
        return (
            f[vol_cols].to_numpy(),
            f["target_rv"].to_numpy(),
            f["gk_vol"].to_numpy(),
        )

    dev_frame = feat_valid.iloc[:dev_end]
    x_dev, y_dev, gk_dev = _slice_xy(dev_frame)

    n = len(y_dev)  # trainable DEV rows (after the dev-only forward-target dropna)
    if n < _MIN_OBS:
        log.warning(
            "skip vol model for %s: only %d trainable dev rows (need %d)",
            symbol,
            n,
            _MIN_OBS,
        )
        return {}, None

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
        x_hold, y_hold, gk_hold = _slice_xy(feat_valid.iloc[dev_end:])
        if len(y_hold) > 0:
            hold_model = make_regressor(n_estimators=200)
            hold_model.fit(x_dev, y_dev)
            hold_preds = hold_model.predict(x_hold)

            ss_res_h = float(np.sum((y_hold - hold_preds) ** 2))
            ss_tot_h = float(np.sum((y_hold - y_hold.mean()) ** 2))
            holdout_oos_r2 = 1.0 - ss_res_h / ss_tot_h if ss_tot_h > 1e-20 else 0.0

            # Same persistence/EWMA baseline as the CV, evaluated within the holdout slice.
            # The EWMA recursion is seeded from the holdout slice's own gk vol (causal, no
            # look-ahead) — the readout is self-contained to the holdout period.
            holdout_base = _walk_forward_ewma_baseline(gk_hold, np.arange(len(gk_hold)))
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
    # The live forecast legitimately uses ALL valid rows (the holdout is an evaluation device,
    # not a data quarantine for the production prediction).  Built on the full series here.
    full = feat_valid.copy()
    full["target_rv"] = _make_targets(full["gk_vol"], horizon=_HORIZON)
    x_full_feat = full[vol_cols].to_numpy()
    full_trained = full.dropna(subset=["target_rv"])
    final = make_regressor(n_estimators=200)
    final.fit(full_trained[vol_cols].to_numpy(), full_trained["target_rv"].to_numpy())
    next_vol = float(final.predict(x_full_feat[[-1]])[0])

    forecast = {
        "symbol": symbol,
        "as_of": pd.to_datetime(feat_valid["date"].iloc[-1]),
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
