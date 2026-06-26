"""Orchestrates the ML layer and persists results back into the marts schema."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mmi.ml import forecast, regime
from mmi.ml.volatility import MODEL_TAG as VOL_MODEL_TAG
from mmi.ml.volatility import train_and_backtest_vol
from mmi.utils.db import init_schemas
from mmi.utils.logging import get_logger

log = get_logger("ml.pipeline")


def _write(con, table: str, df: pd.DataFrame) -> None:
    con.register("_tmp", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp")
    con.unregister("_tmp")


def run_ml(con, symbols: list[str] | None = None) -> dict:
    """Label regimes, backtest + forecast each symbol, write marts.* outputs."""
    init_schemas(con)
    symbols = symbols or ["SPY"]
    now = datetime.now(timezone.utc)

    # 1. Regimes for every asset.
    _write(con, "marts.fct_regime", regime.label_regimes(con))

    # 2. Direction forecast + metrics per requested symbol.
    metric_rows, forecast_rows = [], []
    for sym in symbols:
        try:
            metrics, fc = forecast.train_and_backtest(con, sym)
        except ValueError as exc:
            log.warning("skip %s: %s", sym, exc)
            continue
        forecast_rows.append(fc)

        # Original 5 direction-model metric rows
        for name in ("mae", "baseline_mae", "dir_acc", "baseline_dir_acc", "n_obs"):
            metric_rows.append(
                {
                    "model": "random_forest",
                    "symbol": sym,
                    "metric": name,
                    "value": float(metrics[name]),
                    "trained_at": now,
                }
            )

        # C4: honest secondary skill rows for the direction model
        mae_skill_ratio = (
            metrics["mae"] / metrics["baseline_mae"]
            if metrics["baseline_mae"] > 1e-20
            else float("nan")
        )
        dir_acc_edge = metrics["dir_acc"] - metrics["baseline_dir_acc"]

        for name, value in (
            ("mae_skill_ratio", mae_skill_ratio),
            ("dir_acc_edge", dir_acc_edge),
        ):
            metric_rows.append(
                {
                    "model": "random_forest",
                    "symbol": sym,
                    "metric": name,
                    "value": float(value),
                    "trained_at": now,
                }
            )

        # Locked-holdout rows (honest extra OOS readout; reported, not gated).  Only present
        # when the holdout was carved (enough dev rows); persist whatever keys were emitted.
        for name in ("holdout_dir_acc", "holdout_baseline_dir_acc", "holdout_n_obs"):
            if name in metrics:
                metric_rows.append(
                    {
                        "model": "random_forest",
                        "symbol": sym,
                        "metric": name,
                        "value": float(metrics[name]),
                        "trained_at": now,
                    }
                )

    # 3. HAR realized-vol model (rv_har) per requested symbol.
    for sym in symbols:
        try:
            vol_metrics, vol_fc = train_and_backtest_vol(con, sym)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip vol model for %s: %s", sym, exc)
            continue

        if not vol_metrics:
            # Small-sample skip — train_and_backtest_vol already logged
            continue

        # Persist vol metric rows (Contract D: new rows only, never new columns).
        # The holdout_* rows are an honest extra OOS readout (reported, not gated); they are
        # only present when the locked holdout was carved (enough dev rows), so persist any
        # holdout_* key that the model actually emitted.
        vol_metric_names = [
            "oos_r2",
            "qlike",
            "baseline_qlike",
            "qlike_skill_ratio",
            "n_folds",
            "folds_passed",
            "n_obs",
        ]
        for name in (
            "holdout_oos_r2",
            "holdout_qlike",
            "holdout_qlike_skill_ratio",
            "holdout_n_obs",
        ):
            if name in vol_metrics:
                vol_metric_names.append(name)

        for name in vol_metric_names:
            metric_rows.append(
                {
                    "model": VOL_MODEL_TAG,
                    "symbol": sym,
                    "metric": name,
                    "value": float(vol_metrics[name]),
                    "trained_at": now,
                }
            )

        if vol_fc is not None:
            forecast_rows.append(vol_fc)

    if metric_rows:
        _write(con, "marts.model_metrics", pd.DataFrame(metric_rows))
    if forecast_rows:
        _write(con, "marts.ml_forecast", pd.DataFrame(forecast_rows))

    summary = {f"{r['symbol']}.{r['metric']}": r["value"] for r in metric_rows}
    log.info("ml complete: %s", summary)
    return summary
