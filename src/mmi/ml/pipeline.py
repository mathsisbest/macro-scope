"""Orchestrates the ML layer and persists results back into the marts schema."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mmi.ml import forecast, regime
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

    # 2. Forecast + metrics per requested symbol.
    metric_rows, forecast_rows = [], []
    for sym in symbols:
        try:
            metrics, fc = forecast.train_and_backtest(con, sym)
        except ValueError as exc:
            log.warning("skip %s: %s", sym, exc)
            continue
        forecast_rows.append(fc)
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

    if metric_rows:
        _write(con, "marts.model_metrics", pd.DataFrame(metric_rows))
    if forecast_rows:
        _write(con, "marts.ml_forecast", pd.DataFrame(forecast_rows))

    summary = {f"{r['symbol']}.{r['metric']}": r["value"] for r in metric_rows}
    log.info("ml complete: %s", summary)
    return summary
