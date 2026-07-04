"""Orchestrates the ML layer and persists results back into the marts schema."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mmi.ml import regime
from mmi.ml.forecast import train_and_predict
from mmi.ml.research import _load_asset_data, _load_macro_data
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

    # Load macro/asset data for rich features
    macro_df = _load_macro_data(con)
    asset_dfs = _load_asset_data(con, ["GLD", "TLT"])

    # 1. Regimes for every asset.
    _write(con, "marts.fct_regime", regime.label_regimes(con))

    # 2. Return forecast + metrics per requested symbol.
    metric_rows, forecast_rows = [], []
    for sym in symbols:
        try:
            metrics, forecasts = train_and_predict(
                con,
                symbol=sym,
                feature_set="vol_rich",
                macro_df=macro_df,
                asset_dfs=asset_dfs,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("skip return forecast for %s: %s", sym, exc)
            continue

        # Persist per-horizon metrics
        for horizon, h_metrics in metrics.get("horizons", {}).items():
            for name in (
                "dir_acc",
                "baseline_dir_acc",
                "mae",
                "baseline_mae",
                "r2",
                "ic",
                "n_obs",
            ):
                if name in h_metrics:
                    metric_rows.append(
                        {
                            "model": "return_rf",
                            "symbol": sym,
                            "metric": f"{name}_h{horizon}",
                            "value": float(h_metrics[name]),
                            "trained_at": now,
                        }
                    )
            # Per-regime breakdown
            for regime_name in ("low", "medium", "high"):
                key = f"dir_acc_{regime_name}"
                if key in h_metrics:
                    metric_rows.append(
                        {
                            "model": "return_rf",
                            "symbol": sym,
                            "metric": f"dir_acc_{regime_name}_h{horizon}",
                            "value": float(h_metrics[key]),
                            "trained_at": now,
                        }
                    )
            # Holdout metrics
            for name in (
                "holdout_dir_acc",
                "holdout_baseline_dir_acc",
                "holdout_r2",
                "holdout_ic",
                "holdout_n_obs",
            ):
                if name in h_metrics:
                    metric_rows.append(
                        {
                            "model": "return_rf",
                            "symbol": sym,
                            "metric": f"{name}_h{horizon}",
                            "value": float(h_metrics[name]),
                            "trained_at": now,
                        }
                    )

        # Aggregate metrics
        for name in ("overall_dir_acc", "overall_r2", "overall_ic"):
            if name in metrics:
                metric_rows.append(
                    {
                        "model": "return_rf",
                        "symbol": sym,
                        "metric": name,
                        "value": float(metrics[name]),
                        "trained_at": now,
                    }
                )

        # Persist forecasts (one row per horizon)
        for fc in forecasts:
            forecast_rows.append(fc)

    # 3. HAR realized-vol model (rv_har) per requested symbol.
    for sym in symbols:
        try:
            vol_metrics, vol_fc = train_and_backtest_vol(con, sym)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip vol model for %s: %s", sym, exc)
            continue

        if not vol_metrics:
            continue

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
