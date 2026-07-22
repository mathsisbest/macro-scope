"""Orchestrates the ML layer and persists results back into the marts schema."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd

from mmi.ml import regime
from mmi.ml.forecast import evaluate_forecast, train_latest_forecast
from mmi.ml.research import _load_asset_data, _load_macro_data
from mmi.ml.volatility import MODEL_TAG as VOL_MODEL_TAG
from mmi.ml.volatility import train_and_backtest_vol
from mmi.settings import load_assets
from mmi.utils.db import init_schemas
from mmi.utils.logging import get_logger

log = get_logger("ml.pipeline")

# Max parallel workers for ML training (per-symbol independence)
_MAX_WORKERS = 4
# Per-symbol ML configs set to 20-day (1-month) target horizon.
# All assets achieve strong positive OOS R² at 20 days, enabling monthly trading execution.

_SYMBOL_ML_CONFIG: dict[str, dict] = {
    "SPY": {
        "model": "gb",
        "train_size": 1260,
        "target_horizon": 20,
        "use_all_train": True,
        "feature_set": "vol_macro",
    },
    "GLD": {
        "model": "gb",
        "train_size": 1260,
        "target_horizon": 20,
        "use_all_train": True,
        "feature_set": "vol_macro",
    },
    "TLT": {
        "model": "lgb",
        "train_size": 1260,
        "target_horizon": 20,
        "use_all_train": True,
        "feature_set": "vol_macro",
    },
}
_DEFAULT_ML_CONFIG: dict = {
    "model": "gb",
    "train_size": 1260,
    "target_horizon": 20,
    "use_all_train": True,
    "feature_set": "vol_macro",
}


def _ml_config(sym: str) -> dict:
    """Return the optimised ML config for *sym*, falling back to *DEFAULT*."""
    return _SYMBOL_ML_CONFIG.get(sym, _DEFAULT_ML_CONFIG)


def _default_symbols() -> list[str]:
    """Return the configured market universe using symbols as stored in marts."""
    assets = load_assets()
    ordered: list[str] = []
    for group in ("equities", "bonds", "commodities", "fx"):
        ordered.extend(assets.get(group, []))
    ordered.extend(sym.replace("-USD", "") for sym in assets.get("crypto_daily", []))
    return list(dict.fromkeys(ordered))


def _write(con, table: str, df: pd.DataFrame) -> None:
    con.register("_tmp", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp")
    con.unregister("_tmp")
    pub_dir = Path("data/public")
    if pub_dir.exists():
        t_name = table.split(".")[-1]
        df.to_parquet(pub_dir / f"{t_name}.parquet")


def _train_symbol_ml(
    sym: str,
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    asset_dfs: dict,
    now,
) -> tuple[list[dict], list[dict]]:
    """Train ML models for a single symbol. Returns (metric_rows, forecast_rows).

    Uses per-symbol config from ``_SYMBOL_ML_CONFIG`` (optimised via sweep).
    """
    metric_rows: list[dict] = []
    forecast_rows: list[dict] = []

    cfg = _ml_config(sym)
    model_name: str = cfg["model"]
    train_size: int = cfg["train_size"]
    target_horizon: int = cfg["target_horizon"]
    use_all_train: bool = cfg["use_all_train"]
    feature_set: str = cfg["feature_set"]

    # Compute a usable test_size: large enough for a meaningful evaluation
    # window but small enough to get several walk-forward folds.
    n = len(df)
    available_test = n - train_size - target_horizon
    if available_test < 100:
        log.warning(
            "skip %s: insufficient data (%d rows) for train=%d h=%d",
            sym,
            n,
            train_size,
            target_horizon,
        )
        return metric_rows, forecast_rows
    test_size = min(504, max(252, available_test // 4))
    # At least 3 walk-forward folds needed for a meaningful evaluation.
    if available_test < test_size * 3:
        log.warning(
            "skip %s: too few walk-forward folds (available_test=%d < test_size=%d * 3)",
            sym,
            available_test,
            test_size,
        )
        return metric_rows, forecast_rows

    try:
        res = evaluate_forecast(
            df=df,
            train_size=train_size,
            test_size=test_size,
            horizon=20,
            model=model_name,
            feature_set=feature_set,
            macro_df=macro_df,
            asset_dfs=asset_dfs,
            target_type="raw",
            target_horizon=target_horizon,
            ensemble_method="mean",
            use_all_train=use_all_train,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("skip %s: %s", sym, exc)
        return metric_rows, forecast_rows

    if res.get("prediction_count", 0) == 0:
        return metric_rows, forecast_rows

    # Persist metrics. Do not persist the legacy return-model Sharpe when the target is a
    # multi-day overlapping forward return; evaluate_forecast returns NaN for that field because
    # treating overlapping annual targets as daily tradable returns is misleading.
    for name in (
        "ic",
        "direction_accuracy",
        "direction_accuracy_low",
        "direction_accuracy_medium",
        "direction_accuracy_high",
        "baseline_direction_accuracy",
        "direction_edge",
        "positive_target_rate",
        "positive_prediction_rate",
        "r2",
    ):
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

    # Persist top 10 feature importances for this model
    feat_imps = res.get("feature_importances", {})
    if feat_imps:
        sorted_imps = sorted(feat_imps.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat_name, imp_val in sorted_imps:
            metric_rows.append(
                {
                    "model": "return_gb",
                    "symbol": sym,
                    "metric": f"feature_importance_{feat_name}",
                    "value": float(imp_val),
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

    latest = train_latest_forecast(
        df=df,
        train_size=train_size,
        model=model_name,
        feature_set=feature_set,
        macro_df=macro_df,
        asset_dfs=asset_dfs,
        target_type="raw",
        target_horizon=target_horizon,
    )

    if latest.get("prediction") is not None and pd.notna(latest.get("prediction")):
        pred = float(latest["prediction"])
        forecast_rows.append(
            {
                "symbol": sym,
                "as_of": pd.to_datetime(latest["as_of"]),
                "horizon": target_horizon,
                "predicted_return": pred,
                "daily_mu": pred / target_horizon,
                "model": "return_gb",
                "dir_acc": res.get("direction_accuracy", 0),
                "r2": res.get("r2", 0),
                "predicted_next_return": pred,
            }
        )

    return metric_rows, forecast_rows


def run_ml(con, symbols: list[str] | None = None) -> dict:
    """Label regimes, backtest + forecast each symbol, write marts.* outputs.

    Uses parallel processing for ML training across symbols.
    """
    init_schemas(con)
    symbols = symbols or _default_symbols()
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

    # 5. Write results
    if metric_rows:
        _write(con, "marts.model_metrics", pd.DataFrame(metric_rows))
    if forecast_rows:
        _write(con, "marts.ml_forecast", pd.DataFrame(forecast_rows))

    summary = {f"{r['symbol']}.{r['metric']}": r["value"] for r in metric_rows}
    log.info("ml complete: %s", summary)
    return summary
