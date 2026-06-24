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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from mmi.ml.features import feature_columns, make_features
from mmi.ml.forecast import make_regressor
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

    # --- walk-forward evaluation ---
    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals, base_preds = [], [], []
    for train_idx, test_idx in tscv.split(x):
        model = make_regressor(n_estimators=100)
        model.fit(x[train_idx], y[train_idx])
        preds.append(model.predict(x[test_idx]))
        actuals.append(y[test_idx])
        # EWMA baseline computed walk-forward: only gk vol up to each fold's last test
        # point feeds the recursion — no look-ahead (honest-OOS contract).
        base_preds.append(_walk_forward_ewma_baseline(gk_valid, test_idx))

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
        "n_obs": n,
        "oos_r2": oos_r2,
        "qlike": qlike,
        "baseline_qlike": baseline_qlike,
        "qlike_skill_ratio": qlike_skill_ratio,
        "n_folds": n_folds,
        "folds_passed": folds_passed,
    }

    # --- final model on all data -> forecast next week ---
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
