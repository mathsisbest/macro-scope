"""Point-in-time, horizon-matched walk-forward return forecasts for the ML efficient frontier.

For each rebalance date and asset we forecast the expected forward-``horizon``-day return from a
model fit on **strictly prior, fully-realised** data. The subtlety that makes this look-ahead-free:
the target at day ``d`` spans ``[d+1, d+horizon]``, so the last ``horizon`` rows before a rebalance
must be **embargoed** from training — otherwise their targets would peek at returns on/after the
rebalance date. Features use only data before the rebalance.

We also return walk-forward out-of-sample **skill** per asset (MAE and R^2 vs naive baselines). That
is the *magnitude* evidence the C3 gate consumes: directional accuracy alone does not protect a
mean-variance optimiser from noisy return *magnitudes*, which is the classic ML-MVO footgun.

Note: monthly-horizon targets at ~monthly rebalances overlap little, but any overlap induces serial
correlation that inflates the apparent significance of the skill — the gate must treat it as weak
evidence (handled in C3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmi.ml.features import feature_columns, make_features
from mmi.ml.forecast import make_regressor
from mmi.utils.logging import get_logger

log = get_logger("ml.forecast_panel")


def _skill(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Out-of-sample magnitude + direction skill of predictions vs realised forward returns."""
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "n_preds": int(len(pred)),
        "mae": float(np.mean(np.abs(pred - actual))),
        "baseline_mae": float(np.mean(np.abs(actual))),  # MAE of a predict-zero forecast (mean|y|)
        "r2_oos": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "dir_acc": float(np.mean(np.sign(pred) == np.sign(actual))),
        "baseline_dir_acc": float(max((actual > 0).mean(), (actual <= 0).mean())),
    }


def walk_forward_mu(
    asset_daily: pd.DataFrame,
    rebalance_dates,
    *,
    horizon: int = 21,
    min_train: int = 120,
    n_estimators: int = 100,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-in-time forecast of each asset's forward-``horizon`` return at each rebalance date.

    ``asset_daily``: long ``[symbol, date, daily_return]``. Returns ``(mu_panel, skill)``:
    - ``mu_panel``: long ``[date, symbol, mu]`` — the point-in-time expected return, expressed as a
      **daily-equivalent** (the horizon forecast divided by ``horizon``). This keeps ``mu``
      commensurate with the daily covariance and with the historical-mean ``mu`` the C3 gate blends
      it against (``max_sharpe`` is scale-invariant in ``mu``, but a shrunk-view blend is not).
    - ``skill``: per-symbol **retrospective** walk-forward OOS ``[symbol, horizon, n_preds, mae,
      baseline_mae, r2_oos, dir_acc, baseline_dir_acc]`` — a full-sample diagnostic of forecast
      quality. It is NOT point-in-time; the C3 gate must recompute skill expanding-window so a
      decision at date t uses only outcomes realised before t.
    """
    rebals = sorted(pd.to_datetime(list(rebalance_dates)))
    cols = feature_columns()
    mu_rows: list[dict] = []
    skill_rows: list[dict] = []

    for symbol, group in asset_daily.sort_values("date").groupby("symbol"):
        feats = make_features(group[["date", "daily_return"]])
        # Forward horizon return (arithmetic) realised over [d+1, d+horizon]; NaN near the tail.
        feats["target_h"] = feats["ret"].rolling(horizon).sum().shift(-horizon)

        preds: list[float] = []
        actuals: list[float] = []
        for rebal in rebals:
            hist = feats[feats["date"] < rebal]  # strictly before the rebalance
            if len(hist) <= horizon:
                continue
            # Embargo: drop the last `horizon` rows — their targets reach into [rebal, ...).
            train = hist.iloc[: len(hist) - horizon].dropna(subset=[*cols, "target_h"])
            if len(train) < min_train:
                continue
            x_pred = hist[cols].iloc[[-1]].to_numpy()  # latest features, all known before `rebal`
            if not np.isfinite(x_pred).all():
                continue
            model = make_regressor(n_estimators=n_estimators, seed=seed)
            model.fit(train[cols].to_numpy(), train["target_h"].to_numpy())
            forecast = float(model.predict(x_pred)[0])  # cumulative forward-horizon return
            # Store as a daily-equivalent so mu is commensurate with daily cov / historical-mean mu.
            mu_rows.append({"date": rebal, "symbol": symbol, "mu": forecast / horizon})
            # Realised cumulative forward return (known only later — used for skill, never fitting).
            realised = float(hist["target_h"].iloc[-1])
            if np.isfinite(realised):
                preds.append(forecast)  # skill compares forecast vs realised on the same scale
                actuals.append(realised)

        if preds:
            skill_rows.append(
                {"symbol": symbol, "horizon": horizon, **_skill(np.array(preds), np.array(actuals))}
            )

    log.info("forecast panel: %d mu rows, %d assets scored", len(mu_rows), len(skill_rows))
    return pd.DataFrame(mu_rows), pd.DataFrame(skill_rows)
