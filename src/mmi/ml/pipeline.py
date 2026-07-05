"""Orchestrates the ML layer and persists results back into the marts schema."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mmi.ml import regime
from mmi.ml.forecast import evaluate_forecast
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
        # Load OHLC data from DuckDB
        try:
            df = con.execute(
                "select date, open, high, low, close, daily_return "
                "from marts.fct_asset_daily "
                "where symbol = ? order by date",
                [sym],
            ).df()
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: cannot load data: %s", sym, exc)
            continue

        if df.empty or len(df) < 300:
            log.warning("skip %s: insufficient data (%d rows)", sym, len(df))
            continue

        # Run multi-horizon walk-forward evaluation with GB
        horizons = [5, 10, 20]
        for h in horizons:
            try:
                res = evaluate_forecast(
                    df=df,
                    train_size=250,
                    test_size=20,
                    horizon=h,
                    model="gb",
                    feature_set="vol_rich",
                    macro_df=macro_df,
                    asset_dfs=asset_dfs,
                    target_type="raw",
                    ensemble_method="mean",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("skip %s h=%d: %s", sym, h, exc)
                continue

            if res.get("prediction_count", 0) == 0:
                continue

            # Persist metrics
            for name in ("ic", "direction_accuracy", "r2", "sharpe"):
                val = res.get(name)
                if val is not None and pd.notna(val):
                    metric_rows.append({
                        "model": "return_gb",
                        "symbol": sym,
                        "metric": f"{name}_h{h}",
                        "value": float(val),
                        "trained_at": now,
                    })
            metric_rows.append({
                "model": "return_gb",
                "symbol": sym,
                "metric": f"n_obs_h{h}",
                "value": float(res.get("prediction_count", 0)),
                "trained_at": now,
            })

        # Live forecast: use the latest window's prediction from the best horizon
        # For now, run h=20 as the primary live forecast
        try:
            live_res = evaluate_forecast(
                df=df,
                train_size=250,
                test_size=20,
                horizon=20,
                model="gb",
                feature_set="vol_rich",
                macro_df=macro_df,
                asset_dfs=asset_dfs,
                target_type="raw",
                ensemble_method="mean",
            )
            if live_res.get("prediction_count", 0) > 0:
                last_pred = live_res["predictions"].iloc[-1]
                last_date = live_res["dates"].iloc[-1]
                forecast_rows.append({
                    "symbol": sym,
                    "as_of": pd.to_datetime(last_date),
                    "horizon": 20,
                    "predicted_return": float(last_pred),
                    "daily_mu": float(last_pred) / 20,
                    "model": "return_gb",
                    "dir_acc": live_res.get("direction_accuracy", 0),
                    "r2": live_res.get("r2", 0),
                })
        except Exception as exc:  # noqa: BLE001
            log.warning("skip live forecast for %s: %s", sym, exc)

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
