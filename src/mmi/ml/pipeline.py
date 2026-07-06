"""Orchestrates the ML layer and persists results back into the marts schema."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import cast

import pandas as pd

from mmi.ml import regime
from mmi.ml.forecast import evaluate_forecast
from mmi.ml.research import _load_asset_data, _load_macro_data
from mmi.ml.volatility import MODEL_TAG as VOL_MODEL_TAG
from mmi.ml.volatility import train_and_backtest_vol
from mmi.utils.db import init_schemas
from mmi.utils.logging import get_logger

log = get_logger("ml.pipeline")

# Max parallel workers for ML training (per-symbol independence)
_MAX_WORKERS = 4


def _write(con, table: str, df: pd.DataFrame) -> None:
    con.register("_tmp", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp")
    con.unregister("_tmp")


def _train_symbol_ml(
    sym: str,
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    asset_dfs: dict,
    now,
) -> tuple[list[dict], list[dict]]:
    """Train ML models for a single symbol. Returns (metric_rows, forecast_rows)."""
    metric_rows: list[dict] = []
    forecast_rows: list[dict] = []

    # Single config: train=160, target_horizon=252, vol_macro, test_size=300
    try:
        res = evaluate_forecast(
            df=df,
            train_size=160,
            test_size=300,
            horizon=20,
            model="gb",
            feature_set="vol_macro",
            macro_df=macro_df,
            asset_dfs=asset_dfs,
            target_type="raw",
            target_horizon=252,
            ensemble_method="mean",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("skip %s: %s", sym, exc)
        return metric_rows, forecast_rows

    if res.get("prediction_count", 0) == 0:
        return metric_rows, forecast_rows

    # Persist metrics
    for name in ("ic", "direction_accuracy", "r2", "sharpe"):
        val = res.get(name)
        if val is not None and pd.notna(val):
            metric_rows.append(
                {
                    "model": "return_gb",
                    "symbol": sym,
                    "metric": name,
                    "value": float(val),
                    "trained_at": now,
                }
            )
    metric_rows.append(
        {
            "model": "return_gb",
            "symbol": sym,
            "metric": "n_obs",
            "value": float(res.get("prediction_count", 0)),
            "trained_at": now,
        }
    )

    # Live forecast
    if res.get("prediction_count", 0) > 0:
        last_pred = res["predictions"].iloc[-1]
        last_date = res["dates"].iloc[-1]
        forecast_rows.append(
            {
                "symbol": sym,
                "as_of": pd.to_datetime(last_date),
                "horizon": 252,
                "predicted_return": float(last_pred),
                "daily_mu": float(last_pred) / 252,
                "model": "return_gb",
                "dir_acc": res.get("direction_accuracy", 0),
                "r2": res.get("r2", 0),
            }
        )

    return metric_rows, forecast_rows


def run_ml(con, symbols: list[str] | None = None) -> dict:
    """Label regimes, backtest + forecast each symbol, write marts.* outputs.

    Uses parallel processing for ML training across symbols.
    """
    init_schemas(con)
    symbols = symbols or ["SPY"]
    now = datetime.now(timezone.utc)

    # Load macro/asset data for rich features
    macro_df = _load_macro_data(con)
    asset_dfs = _load_asset_data(con, ["GLD", "TLT"])

    # 1. Regimes for every asset.
    _write(con, "marts.fct_regime", regime.label_regimes(con))

    # 2. Load data for all symbols
    symbol_data = {}
    for sym in symbols:
        try:
            df = con.execute(
                "select date, open, high, low, close, daily_return "
                "from marts.fct_asset_daily "
                "where symbol = ? order by date",
                [sym],
            ).df()
            if df.empty or len(df) < 300:
                log.warning("skip %s: insufficient data (%d rows)", sym, len(df))
                continue
            symbol_data[sym] = df
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: cannot load data: %s", sym, exc)

    # 3. Parallel ML training across symbols
    metric_rows: list[dict] = []
    forecast_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_train_symbol_ml, sym, df, macro_df, asset_dfs, now): sym
            for sym, df in symbol_data.items()
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                m_rows, f_rows = future.result()
                metric_rows.extend(m_rows)
                forecast_rows.extend(f_rows)
                log.info(
                    "completed ML for %s: %d metrics, %d forecasts", sym, len(m_rows), len(f_rows)
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("ML failed for %s: %s", sym, exc)

    # 4. HAR realized-vol model (rv_har) per symbol — also parallel
    def _train_vol(sym):
        try:
            vol_metrics, vol_fc = train_and_backtest_vol(con, sym)
            return sym, vol_metrics, vol_fc
        except Exception as exc:  # noqa: BLE001
            log.warning("skip vol model for %s: %s", sym, exc)
            return sym, None, None

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        vol_futures: dict = {executor.submit(_train_vol, sym): sym for sym in symbols}
        for future in as_completed(vol_futures):
            sym, vol_metrics, vol_fc = cast(tuple, future.result())
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

    # 5. Write results
    if metric_rows:
        _write(con, "marts.model_metrics", pd.DataFrame(metric_rows))
    if forecast_rows:
        _write(con, "marts.ml_forecast", pd.DataFrame(forecast_rows))

    summary = {f"{r['symbol']}.{r['metric']}": r["value"] for r in metric_rows}
    log.info("ml complete: %s", summary)
    return summary
