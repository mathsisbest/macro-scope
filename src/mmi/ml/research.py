"""ML research sweep — systematic comparison of model × features × horizon configs.

Run as a script: ``python -m mmi.ml.research``
Or import and call ``run_research(con)``.

The sweep is READ-ONLY against the DB — it never writes to model_metrics or ml_forecast.
Results are printed to stdout and returned as a DataFrame for manual review.
"""

from __future__ import annotations

import io
import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mmi.ml.forecast import evaluate_forecast
from mmi.ml.metrics import ForecastEvaluationResult
from mmi.ml.research_forecast import _load_asset_vol, _pivot_macro, _spy_df
from mmi.ml.skill_gate import return_forecast_skill_verdict, skill_verdict
from mmi.ml.volatility import train_and_backtest_vol
from mmi.settings import REPO_ROOT
from mmi.utils.db import connect
from mmi.utils.logging import get_logger

log = get_logger("ml.research")

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

MODELS = ["rv_har", "rv_ridge", "rv_lasso", "rv_gb", "rv_har_regime"]
FEATURE_SETS = ["vol", "vol_macro", "vol_rich"]
HORIZONS = [5, 10]
N_SPLITS_LIST = [5, 10]

# Ridge / Lasso alpha grid (only used for those models)
_ALPHA_GRID = [0.01, 0.1, 1.0]

# GB hyperparameter grid
_GB_GRID = [
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 3},
]


def _model_params(model_name: str) -> list[dict]:
    """Return the hyperparameter combos to sweep for a given model."""
    if model_name == "rv_har":
        return [{}]  # OLS has no hyperparams
    if model_name == "rv_ridge":
        return [{"alpha": a} for a in _ALPHA_GRID]
    if model_name == "rv_lasso":
        return [{"alpha": a} for a in _ALPHA_GRID]
    if model_name == "rv_gb":
        return list(_GB_GRID)
    return [{}]


def _load_macro_data(con) -> pd.DataFrame:
    """Load FRED series + Shiller CAPE, pivoted to a wide daily DataFrame.

    Returns one row per date with a column per series_id plus ``cape`` and
    ``excess_cape_yield``.  The feature builder ASOF-merges this onto trading dates.
    """
    try:
        df = con.execute(
            "select date, series_id, value from marts.fct_macro_indicator order by date"
        ).df()
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot_table(index="date", columns="series_id", values="value", aggfunc="first")
        wide = wide.reset_index().sort_values("date")
        for col in wide.columns:
            if col != "date":
                wide[col] = wide[col].ffill()

        # Merge Shiller CAPE data — monthly, ASOF-merged onto daily grid
        cape_df = _load_cape_data()
        if not cape_df.empty:
            wide["date"] = pd.to_datetime(wide["date"])
            cape_df["date"] = pd.to_datetime(cape_df["date"])
            wide = pd.merge_asof(
                wide.sort_values("date"),
                cape_df.sort_values("date"),
                on="date",
                direction="backward",
            )
            # Forward-fill CAPE-derived features (monthly) to daily dates
            for col in ["cape", "excess_cape_yield", "div_yield", "earn_yield"]:
                if col in wide.columns:
                    wide[col] = wide[col].ffill()

        return wide
    except Exception:
        return pd.DataFrame()


def _load_cape_data() -> pd.DataFrame:
    """Download Shiller CAPE data and return as a (date, cape, excess_cape_yield) DataFrame.

    The source is Robert Shiller's public spreadsheet at Yale.  Data is monthly from 1881.
    Also extracts dividend yield and earnings yield computed from raw P/D/E columns.
    Returns an empty DataFrame if download or parse fails.
    """
    import urllib.request

    cache_path = REPO_ROOT / "data/raw/shiller_cape.parquet"
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read()
        df = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None, skiprows=8)
    except Exception:
        log.warning("Failed to download Shiller CAPE data — skipping")
        return pd.DataFrame()

    # Columns: 0=date, 1=P, 2=D, 3=E, 12=CAPE, 16=excess_CAPE_yield
    out = df[[0, 1, 2, 3, 12, 16]].copy()
    out.columns = ["date_str", "P", "D", "E", "cape", "excess_cape_yield"]

    def _parse(d):
        if pd.isna(d):
            return pd.NaT
        parts = str(d).split(".")
        return pd.Timestamp(
            year=int(parts[0]),
            month=int(parts[1]) if len(parts) > 1 else 1,
            day=1,
        )

    out["date"] = out["date_str"].apply(_parse)
    for col in ["P", "D", "E", "cape", "excess_cape_yield"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Compute yields: dividend yield and earnings yield as decimals (not %)
    out["div_yield"] = out["D"] / out["P"]
    out["earn_yield"] = out["E"] / out["P"]

    out = out.dropna(subset=["cape"])
    cols = ["date", "cape", "excess_cape_yield", "div_yield", "earn_yield"]
    out = out[cols].sort_values("date").reset_index(drop=True)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path)
    except Exception:
        pass

    return out


def _load_asset_data(con, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load per-symbol daily data for cross-asset vol features."""
    out = {}
    for sym in symbols:
        try:
            df = con.execute(
                "select date, daily_return from marts.fct_asset_daily "
                "where symbol = ? order by date",
                [sym],
            ).df()
            if not df.empty:
                out[sym] = df
        except Exception:
            pass
    return out


def run_research(
    con,
    symbol: str = "SPY",
    models: list[str] | None = None,
    feature_sets: list[str] | None = None,
    horizons: list[int] | None = None,
    n_splits_list: list[int] | None = None,
) -> pd.DataFrame:
    """Run the full combinatorial sweep and return a comparison DataFrame.

    Parameters
    ----------
    con:
        DuckDB connection.
    symbol:
        Asset to model (default SPY).
    models:
        Subset of MODELS to sweep (default: all).
    feature_sets:
        Subset of FEATURE_SETS to sweep (default: all).
    horizons:
        Subset of HORIZONS to sweep (default: all).
    n_splits_list:
        Subset of N_SPLITS_LIST to sweep (default: all).
    """
    models = models or MODELS
    feature_sets = feature_sets or FEATURE_SETS
    horizons = horizons or HORIZONS
    n_splits_list = n_splits_list or N_SPLITS_LIST

    macro_df = _load_macro_data(con)
    asset_dfs = _load_asset_data(con, ["GLD", "TLT"])

    results: list[dict] = []
    combos = list(itertools.product(models, feature_sets, horizons, n_splits_list))
    total = len(combos)

    for i, (model_name, feat_set, horizon, n_splits) in enumerate(combos, 1):
        # Skip vol_macro if macro data unavailable
        if feat_set == "vol_macro" and macro_df.empty:
            log.info("skip %s/%s: no macro data", model_name, feat_set)
            continue

        params_list = _model_params(model_name)
        for params in params_list:
            param_str = str(params) if params else ""
            log.info(
                "[%d/%d] %s feat=%s h=%d splits=%d %s",
                i,
                total,
                model_name,
                feat_set,
                horizon,
                n_splits,
                param_str,
            )

            try:
                metrics, _forecast = train_and_backtest_vol(
                    con,
                    symbol=symbol,
                    model_name=model_name,
                    feature_set=feat_set,
                    horizon=horizon,
                    n_splits=n_splits,
                    model_params=params if params else None,
                    macro_df=macro_df if feat_set in ("vol_macro", "vol_rich") else None,
                    asset_dfs=asset_dfs if feat_set in ("vol_macro", "vol_rich") else None,
                )
            except Exception as e:
                log.warning("FAILED %s/%s: %s", model_name, feat_set, e)
                continue

            if not metrics:
                log.info("skip %s/%s: small sample", model_name, feat_set)
                continue

            results.append(
                {
                    "model": model_name,
                    "feature_set": feat_set,
                    "horizon": horizon,
                    "n_splits": n_splits,
                    "params": param_str,
                    "oos_r2": metrics.get("oos_r2"),
                    "qlike_skill_ratio": metrics.get("qlike_skill_ratio"),
                    "folds_passed": metrics.get("folds_passed"),
                    "n_folds": metrics.get("n_folds"),
                    "n_obs": metrics.get("n_obs"),
                    "holdout_oos_r2": metrics.get("holdout_oos_r2"),
                    "holdout_qlike_skill_ratio": metrics.get("holdout_qlike_skill_ratio"),
                }
            )

    if not results:
        log.warning("no results — all combos failed or were skipped")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by OOS R² descending (best first)
    df = df.sort_values("oos_r2", ascending=False).reset_index(drop=True)

    # Print summary
    print("\n" + "=" * 80)
    print("ML RESEARCH SWEEP RESULTS")
    print("=" * 80)
    print(f"Symbol: {symbol} | Combos tested: {len(df)}")
    print()

    # Skill gate thresholds
    R2_MIN = 0.10
    QLIKE_MAX = 0.99

    for _, row in df.iterrows():
        cleared = (
            row["oos_r2"] is not None
            and row["oos_r2"] >= R2_MIN
            and row["qlike_skill_ratio"] is not None
            and row["qlike_skill_ratio"] < QLIKE_MAX
            and row["folds_passed"] is not None
            and row["folds_passed"] >= 3
        )
        tag = " CLEARED" if cleared else ""
        print(
            f"  {row['model']:12s} | {row['feature_set']:10s} | h={row['horizon']} "
            f"| splits={row['n_splits']} | R²={row['oos_r2']:.4f} "
            f"| QR={row['qlike_skill_ratio']:.4f} "
            f"| folds={row['folds_passed']}/{row['n_folds']}{tag}"
        )

    print()
    best = df.iloc[0]
    print(
        f"BEST: {best['model']} / {best['feature_set']} / h={best['horizon']} "
        f"/ R²={best['oos_r2']:.4f} / QR={best['qlike_skill_ratio']:.4f}"
    )
    print("=" * 80 + "\n")

    return df


# ---------------------------------------------------------------------------
# Robustness / sensitivity analysis
# ---------------------------------------------------------------------------
# Honesty contract (read before use):
#   * This analysis is DIAGNOSTIC ONLY.  It is NOT a config selector: it never
#     writes to model_metrics / ml_forecast, never feeds the skill gate, and its
#     output must never be used to cherry-pick a config that passes the gate.
#   * Pre-register the base config and the perturbations BEFORE running and look
#     at results once.  Refining perturbations until a config "looks robust" is
#     the same disease as tuning hyperparameters until the gate clears.
#   * Fragility flags are findings, not tuning targets.  A fragile result tells
#     you the config is not trustworthy; it does not tell you which config to
#     deploy.

# Metrics reported per task.  OOS R², IC and direction accuracy are the headline
# skill readouts; QLIKE skill ratio and folds_passed are the vol-model gate inputs.
_TASK_METRICS: dict[str, tuple[str, ...]] = {
    "vol": ("oos_r2", "qlike_skill_ratio", "folds_passed"),
    "return": ("ic", "direction_accuracy", "r2"),
}

# Reference ("no-skill") value per metric — a sign flip crosses this value:
# 0 for R²/IC (vs zero skill), 0.5 for direction accuracy (vs coin flip) and
# 1.0 for the QLIKE skill ratio (model loss = baseline loss).
_METRIC_REFERENCES: dict[str, float] = {
    "oos_r2": 0.0,
    "r2": 0.0,
    "ic": 0.0,
    "direction_accuracy": 0.5,
    "qlike_skill_ratio": 1.0,
}

# Default base configs per task (documented knobs of each runner).
_VOL_BASE_CONFIG: dict = {
    "model_name": "rv_har",
    "feature_set": "vol",
    "horizon": 5,
    "n_splits": 5,
    "min_dev": 60,
    "model_params": None,
}

_RETURN_BASE_CONFIG: dict = {
    "model": "gb",
    "feature_set": "default",
    "target_type": "raw",
    "train_size": 250,
    "test_size": 20,
    "target_horizon": 1,
    "use_all_train": False,
    "model_kwargs": None,
}


@dataclass(frozen=True)
class Perturbation:
    """A declared sensitivity probe: one config knob plus the values to try.

    ``dimension`` names the knob (see :func:`run_robustness_analysis` for the
    per-task mapping) and ``value_options`` are the alternate knob values to
    run.  The base config itself is always run once as the reference; values
    equal to the base knob value are skipped by the runner.
    """

    dimension: str
    value_options: tuple[int | float | bool | str, ...]


def default_perturbations(base_config: dict, task: str = "vol") -> list[Perturbation]:
    """Pre-registered perturbation grid around ``base_config``.

    Honesty contract: these defaults are FIXED functions of the base config —
    symmetric ± steps around the base value.  They are not chosen by looking at
    results and should be declared before the run, never refined to make a
    config look robust.

    ``vol`` task knobs (mapped to :func:`mmi.ml.volatility.train_and_backtest_vol`):
        ``horizon``      target horizon ± 2 trading days (floored at 1)
        ``min_periods``  ``min_dev`` minimum trainable rows ± 20 (floored at 1)
        ``window``       walk-forward ``n_splits`` ± 2 (floored at 2)
        ``seed``         GB ``random_state`` in {1, 2} — rv_gb only, the sole
                         stochastic model (OLS/LASSO/Ridge/HAR are deterministic)

    ``return`` task knobs (mapped to :func:`mmi.ml.forecast.evaluate_forecast`):
        ``train_size``   rolling train window ± 100 rows (floored at 100)
        ``horizon``      ``target_horizon`` ± 1 day (floored at 1)
        ``window``       window strategy: expanding (``use_all_train=True``) vs
                         the base rolling window
        ``seed``         model ``random_state`` in {41, 43}
    """
    if task == "vol":
        horizon = int(base_config.get("horizon", 5))
        min_dev = int(base_config.get("min_dev", 60))
        n_splits = int(base_config.get("n_splits", 5))
        probes = [
            Perturbation("horizon", _dedupe_vs_base((max(1, horizon - 2), horizon + 2), horizon)),
            Perturbation(
                "min_periods", _dedupe_vs_base((max(1, min_dev - 20), min_dev + 20), min_dev)
            ),
            Perturbation("window", _dedupe_vs_base((max(2, n_splits - 2), n_splits + 2), n_splits)),
        ]
        if base_config.get("model_name") == "rv_gb":
            probes.append(Perturbation("seed", (1, 2)))
        return probes
    if task == "return":
        train_size = int(base_config.get("train_size", 250))
        horizon = int(base_config.get("target_horizon", 1))
        return [
            Perturbation(
                "train_size",
                _dedupe_vs_base((max(100, train_size - 100), train_size + 100), train_size),
            ),
            Perturbation("horizon", _dedupe_vs_base((max(1, horizon - 1), horizon + 1), horizon)),
            Perturbation(
                "window",
                _dedupe_vs_base((True,), bool(base_config.get("use_all_train", False))),
            ),
            Perturbation("seed", (41, 43)),
        ]
    raise ValueError(f"unknown task {task!r} — expected 'vol' or 'return'")


def _dedupe_vs_base(
    values: tuple[int | float | bool | str, ...], base: int | float | bool | str
) -> tuple[int | float | bool | str, ...]:
    """Drop probe values that equal the base knob value (they add no information)."""
    return tuple(v for v in values if v != base)


def _base_knob_value(config: dict, task: str, dimension: str) -> int | float | bool | str:
    """The base config's value for a perturbation dimension."""
    if task == "vol":
        if dimension == "horizon":
            return int(config.get("horizon", 5))
        if dimension == "min_periods":
            return int(config.get("min_dev", 60))
        if dimension == "window":
            return int(config.get("n_splits", 5))
        if dimension == "seed":
            return int((config.get("model_params") or {}).get("random_state", 0))
    elif task == "return":
        if dimension == "train_size":
            return int(config.get("train_size", 250))
        if dimension == "horizon":
            return int(config.get("target_horizon", 1))
        if dimension == "window":
            return bool(config.get("use_all_train", False))
        if dimension == "seed":
            return int((config.get("model_kwargs") or {}).get("random_state", 42))
    raise ValueError(f"unknown perturbation dimension {dimension!r} for task {task!r}")


def _perturbation_overrides(
    config: dict, task: str, dimension: str, value: int | float | bool | str
) -> dict:
    """Map a (dimension, value) probe to the kwargs it changes for the task's runner."""
    if task == "vol":
        if dimension == "horizon":
            return {"horizon": int(value)}
        if dimension == "min_periods":
            return {"min_dev": int(value)}
        if dimension == "window":
            return {"n_splits": int(value)}
        if dimension == "seed":
            if config.get("model_name") != "rv_gb":
                raise ValueError(
                    "seed perturbation is only meaningful for model 'rv_gb' — "
                    "OLS/LASSO/Ridge/HAR are deterministic "
                    f"(got model_name={config.get('model_name')!r})"
                )
            params = dict(config.get("model_params") or {})
            params["random_state"] = int(value)
            return {"model_params": params}
    elif task == "return":
        if dimension == "train_size":
            return {"train_size": int(value)}
        if dimension == "horizon":
            return {"target_horizon": int(value)}
        if dimension == "window":
            return {"use_all_train": bool(value)}
        if dimension == "seed":
            return {"model_kwargs": {"random_state": int(value)}}
    raise ValueError(f"unknown perturbation dimension {dimension!r} for task {task!r}")


def _vol_gate_cleared(metrics: dict, model_name: str, symbol: str) -> bool | None:
    """Skill-gate outcome for one vol run, or ``None`` when not applicable.

    The gate (:func:`mmi.ml.skill_gate.skill_verdict`) is scoped to ``rv_har``;
    other vol models are never gated, so they report ``None``.  An empty metric
    dict (small-sample skip) also reports ``None`` — a skipped run is not a
    gate failure.
    """
    if not metrics or model_name != "rv_har":
        return None
    rows = [
        {"model": "rv_har", "symbol": symbol, "metric": m, "value": metrics[m]}
        for m in ("oos_r2", "qlike_skill_ratio", "folds_passed", "n_folds", "n_obs")
        if m in metrics
    ]
    return skill_verdict(pd.DataFrame(rows), symbol=symbol)["cleared"]


def _return_gate_cleared(res: ForecastEvaluationResult, model: str, symbol: str) -> bool | None:
    """Skill-gate outcome for one return-forecast run, or ``None`` when skipped.

    ``res`` is a :class:`mmi.ml.metrics.ForecastEvaluationResult` from
    :func:`mmi.ml.forecast.evaluate_forecast`.  A run with zero predictions is
    a skipped run, not a gate failure → ``None``.
    """
    if res["prediction_count"] == 0:
        return None
    rows = [
        {"model": model, "symbol": symbol, "metric": "r2", "value": res["r2"]},
        {
            "model": model,
            "symbol": symbol,
            "metric": "direction_accuracy",
            "value": res["direction_accuracy"],
        },
        {"model": model, "symbol": symbol, "metric": "folds_passed", "value": res["folds_passed"]},
        {"model": model, "symbol": symbol, "metric": "n_folds", "value": res["n_folds"]},
        {"model": model, "symbol": symbol, "metric": "n_obs", "value": res["prediction_count"]},
        {
            "model": model,
            "symbol": symbol,
            "metric": "annualised_alpha",
            "value": res["annualised_alpha"],
        },
        {
            "model": model,
            "symbol": symbol,
            "metric": "turnover_adjusted_sharpe",
            "value": res["turnover_adjusted_sharpe"],
        },
    ]
    return return_forecast_skill_verdict(pd.DataFrame(rows), symbol=symbol, model=model)["cleared"]


def _run_config(con, symbol: str, task: str, config: dict) -> tuple[dict, bool | None]:
    """Evaluate one config and return ``(metrics, gate_cleared)``.

    ``metrics`` is a flat dict of the task's headline metrics (empty dict when
    the run is skipped); ``gate_cleared`` is the skill-gate outcome for the run
    or ``None`` when the gate does not apply (model not gated / run skipped).
    """
    if task == "vol":
        macro_df = _load_macro_data(con)
        asset_dfs = _load_asset_data(con, ["GLD", "TLT"])
        model_name = config.get("model_name", "rv_har")
        feature_set = config.get("feature_set", "vol")
        needs_macro = feature_set in ("vol_macro", "vol_rich")
        if needs_macro and macro_df.empty:
            return {}, None
        metrics, _forecast = train_and_backtest_vol(
            con,
            symbol=symbol,
            model_name=model_name,
            feature_set=feature_set,
            horizon=int(config.get("horizon", 5)),
            n_splits=int(config.get("n_splits", 5)),
            min_dev=int(config.get("min_dev", 60)),
            model_params=config.get("model_params") or None,
            macro_df=macro_df if needs_macro else None,
            asset_dfs=asset_dfs if needs_macro else None,
        )
        return metrics, _vol_gate_cleared(metrics, model_name, symbol)

    if task == "return":
        df = _spy_df(con)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        macro_df = _pivot_macro(con)
        asset_dfs = _load_asset_vol(con, ("GLD", "TLT"))
        feature_set = config.get("feature_set", "default")
        needs_macro = feature_set in ("vol", "vol_macro", "vol_medium", "vol_rich")
        target_horizon = int(config.get("target_horizon", 1))
        res = evaluate_forecast(
            df=df,
            train_size=int(config.get("train_size", 250)),
            test_size=int(config.get("test_size", 20)),
            horizon=target_horizon,
            target_horizon=target_horizon,
            model=config.get("model", "gb"),
            feature_set=feature_set,
            macro_df=macro_df if needs_macro else None,
            asset_dfs=asset_dfs if needs_macro else None,
            target_type=config.get("target_type", "raw"),
            use_all_train=bool(config.get("use_all_train", False)),
            single_split=False,
            **dict(config.get("model_kwargs") or {}),
        )
        return (
            {m: res[m] for m in _TASK_METRICS["return"]},
            _return_gate_cleared(res, config.get("model", "gb"), symbol),
        )

    raise ValueError(f"unknown task {task!r} — expected 'vol' or 'return'")


def run_robustness_analysis(
    con,
    symbol: str = "SPY",
    task: str = "vol",
    base_config: dict | None = None,
    perturbations: list[Perturbation] | None = None,
) -> pd.DataFrame:
    """Run a robustness / sensitivity analysis around a model config.

    The base config is evaluated once, then each perturbation value is
    evaluated once, and the headline metrics (OOS R², IC, direction accuracy,
    QLIKE skill ratio, folds_passed — per task) plus the skill-gate outcome
    are reported per run.  :func:`aggregate_robustness`,
    :func:`classify_robustness` and :func:`summarize_robustness` turn the raw
    runs into the fragility report.

    Honesty contract — read before use:
      * DIAGNOSTIC ONLY.  This never writes to ``model_metrics`` /
        ``ml_forecast``, never feeds the skill gate, and must never be used to
        cherry-pick a config that passes the gate.
      * Pre-register ``base_config`` and ``perturbations`` BEFORE running and
        read the output once.  Iterating on perturbations until the report says
        "robust" is the same disease as tuning hyperparameters until the gate
        clears.

    Parameters
    ----------
    con:
        DuckDB connection.
    symbol:
        Asset to model (default SPY).
    task:
        ``'vol'`` (realized-volatility research, metrics ``oos_r2`` /
        ``qlike_skill_ratio`` / ``folds_passed``) or ``'return'`` (return
        forecast research, metrics ``ic`` / ``direction_accuracy`` / ``r2``).
    base_config:
        Knobs of the config under test.  ``vol`` keys: ``model_name``,
        ``feature_set``, ``horizon``, ``n_splits``, ``min_dev``,
        ``model_params``.  ``return`` keys: ``model``, ``feature_set``,
        ``target_type``, ``train_size``, ``test_size``, ``target_horizon``,
        ``use_all_train``, ``model_kwargs``.  Defaults match the sweep defaults.
    perturbations:
        Probes to run.  Dimensions (vol): ``horizon``, ``min_periods`` (=
        ``min_dev``), ``window`` (= ``n_splits``), ``seed`` (rv_gb only).
        Dimensions (return): ``train_size``, ``horizon`` (=
        ``target_horizon``), ``window`` (= ``use_all_train``), ``seed``.
        Default: :func:`default_perturbations` for the base config.

    Returns
    -------
    DataFrame
        Long-format runs: one row per (run × metric) with columns ``task``,
        ``symbol``, ``dimension`` (``'base'`` for the reference run), ``value``
        (perturbed knob value, ``None`` for the base run), ``metric``,
        ``metric_value``, ``gate_cleared``.
    """
    if task not in _TASK_METRICS:
        raise ValueError(f"unknown task {task!r} — expected one of {sorted(_TASK_METRICS)}")
    defaults = _VOL_BASE_CONFIG if task == "vol" else _RETURN_BASE_CONFIG
    config = {**defaults, **(base_config or {})}
    if perturbations is None:
        perturbations = default_perturbations(config, task=task)

    # Validate every probe up front (fail fast, honestly) and drop probes whose
    # value equals the base knob value — they add no information.
    probes: list[tuple[str, int | float | bool | str, dict]] = []
    for probe in perturbations:
        for value in probe.value_options:
            overrides = _perturbation_overrides(config, task, probe.dimension, value)
            if value == _base_knob_value(config, task, probe.dimension):
                log.info("robustness: skip %s=%s — equals base value", probe.dimension, value)
                continue
            probes.append((probe.dimension, value, overrides))

    rows: list[dict] = []

    def _emit(
        dimension: str,
        value: int | float | bool | str | None,
        metrics: dict,
        gate_cleared: bool | None,
    ) -> None:
        for metric in _TASK_METRICS[task]:
            rows.append(
                {
                    "task": task,
                    "symbol": symbol,
                    "dimension": dimension,
                    "value": value,
                    "metric": metric,
                    "metric_value": metrics.get(metric, float("nan")),
                    "gate_cleared": gate_cleared,
                }
            )

    base_metrics, base_gate = _run_config(con, symbol, task, config)
    if not base_metrics:
        log.warning("robustness analysis: base config produced no metrics")
    _emit("base", None, base_metrics, base_gate)

    for dimension, value, overrides in probes:
        run_config = {**config, **overrides}
        try:
            metrics, gate_cleared = _run_config(con, symbol, task, run_config)
        except Exception as e:
            log.warning("robustness probe %s=%s failed: %s", dimension, value, e)
            continue
        _emit(dimension, value, metrics, gate_cleared)

    if not rows:
        log.warning("robustness analysis produced no rows")
        return pd.DataFrame()
    return pd.DataFrame(rows)


def aggregate_robustness(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-(dimension, metric) summary of a robustness run.

    The base run (``dimension == 'base'``) defines the reference; probe rows
    are summarised with min / max / mean and ``spread`` — the largest
    ``|probe − base|`` swing across the dimension's valid probes.  All values
    are ``NaN`` when there is nothing to aggregate (missing base or no valid
    probes).
    """
    out: list[dict] = []
    if runs.empty:
        return pd.DataFrame(out)
    base_rows = runs[runs["dimension"] == "base"]
    base_map: dict[str, float] = {}
    if not base_rows.empty:
        for _, r in base_rows.dropna(subset=["metric_value"]).iterrows():
            base_map[r["metric"]] = float(r["metric_value"])
    for (dimension, metric), grp in runs.groupby(["dimension", "metric"]):
        if dimension == "base":
            continue
        probe_rows = grp.dropna(subset=["metric_value"])
        base_val = base_map.get(metric, float("nan"))
        spread = float("nan")
        if not base_rows.empty and not probe_rows.empty:
            spread = float(np.nanmax(np.abs(probe_rows["metric_value"] - base_val)))
        out.append(
            {
                "dimension": dimension,
                "metric": metric,
                "base": base_val,
                "min": _safe_agg(np.nanmin, probe_rows),
                "max": _safe_agg(np.nanmax, probe_rows),
                "mean": _safe_agg(np.nanmean, probe_rows),
                "spread": spread,
                "n_probes": len(probe_rows),
            }
        )
    return pd.DataFrame(out)


def _safe_agg(func, probe_rows: pd.DataFrame) -> float:
    """Apply a numpy nan-agg to probe values, returning NaN when empty."""
    if probe_rows.empty:
        return float("nan")
    return float(func(probe_rows["metric_value"]))


def classify_robustness(
    runs: pd.DataFrame,
    *,
    swing_tolerance: float = 0.05,
    min_probes: int = 1,
) -> pd.DataFrame:
    """Classify each (dimension, metric) pair as ``robust`` / ``fragile`` / ``inconclusive``.

    Flags (each is a red flag measured against the base run):

    ``sign_flip``
        A probe crossed the metric's no-skill reference (0 for R²/IC, 0.5 for
        direction accuracy, 1.0 for the QLIKE skill ratio) — the result is
        sensitive to an innocent config choice.
    ``large_swing``
        A probe moved more than ``swing_tolerance`` away from the base value.
    ``gate_flip``
        The skill gate cleared in base but failed under a probe, or vice versa
        (dimension-level; repeated on each metric row of the dimension).
    ``no_base`` / ``no_probes``
        Not enough information → the pair is inconclusive.

    Precedence: any hard flag (sign flip / large swing / gate flip) ⇒
    ``fragile``; otherwise a missing base or fewer than ``min_probes`` valid
    probes ⇒ ``inconclusive``; otherwise ``robust``.
    """
    out: list[dict] = []
    if runs.empty:
        return pd.DataFrame(out)
    dimensions = sorted(d for d in runs["dimension"].unique() if d != "base")
    base_rows = runs[runs["dimension"] == "base"]
    has_base_run = not base_rows.empty
    base_gate = base_rows["gate_cleared"].iloc[0] if has_base_run else None

    for dimension in dimensions:
        grp = runs[runs["dimension"] == dimension]
        probes = grp.dropna(subset=["metric_value"])
        metrics = sorted(grp["metric"].unique())

        gate_flips = 0
        if base_gate is not None:
            gate_flips = sum(
                1 for gc in grp["gate_cleared"].tolist() if gc is not None and gc != base_gate
            )

        base_metric_map: dict[str, float] = {}
        if has_base_run:
            for _, r in base_rows[base_rows["metric"].isin(metrics)].iterrows():
                if not pd.isna(r["metric_value"]):
                    base_metric_map[r["metric"]] = float(r["metric_value"])

        for metric in metrics:
            flags: list[str] = []
            if gate_flips:
                flags.append("gate_flip")
            if metric not in base_metric_map:
                flags.append("no_base")
            else:
                base_val = base_metric_map[metric]
                ref = _METRIC_REFERENCES.get(metric)
                n_valid = 0
                for _, r in probes[probes["metric"] == metric].iterrows():
                    pv = float(r["metric_value"])
                    n_valid += 1
                    if ref is not None and (base_val - ref) * (pv - ref) < 0:
                        flags.append("sign_flip")
                        break
                    if abs(pv - base_val) > swing_tolerance:
                        flags.append("large_swing")
                        break
                if n_valid < min_probes:
                    flags.append("no_probes")

            hard_flags = {"sign_flip", "large_swing", "gate_flip"}
            if any(f in hard_flags for f in flags):
                verdict = "fragile"
            elif "no_base" in flags or "no_probes" in flags:
                verdict = "inconclusive"
            else:
                verdict = "robust"

            out.append(
                {
                    "dimension": dimension,
                    "metric": metric,
                    "n_probes": len(probes[probes["metric"] == metric]),
                    "flags": flags,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(out)


def summarize_robustness(
    runs: pd.DataFrame,
    *,
    swing_tolerance: float = 0.05,
    as_json: bool = False,
) -> dict:
    """Print a compact robustness table (and optionally JSON) and return both.

    Console/JSON output is consistent with the other research tooling.  The
    honesty framing is printed with the table: the analysis is DIAGNOSTIC ONLY
    — it is not a config selector, nothing here feeds the skill gate, and the
    output must never be used to cherry-pick a config that passes the gate.

    Returns a dict with ``task``, ``symbol``, ``summary`` (merged
    aggregate/classify table rows) and ``overall`` verdict
    (``robust`` / ``fragile`` / ``inconclusive``).
    """
    summary = aggregate_robustness(runs)
    verdicts = classify_robustness(runs, swing_tolerance=swing_tolerance)
    if not summary.empty and not verdicts.empty:
        verdicts = verdicts.drop(columns=["n_probes"])  # keep aggregate's count
        table = summary.merge(verdicts, on=["dimension", "metric"], how="left")
    else:
        table = pd.DataFrame()

    overall = "inconclusive"
    if not table.empty:
        verdicts_list = table["verdict"].tolist()
        if any(v == "fragile" for v in verdicts_list):
            overall = "fragile"
        elif all(v == "robust" for v in verdicts_list):
            overall = "robust"

    print("\n" + "=" * 80)
    print("ROBUSTNESS SENSITIVITY ANALYSIS")
    print("=" * 80)
    task = runs["task"].iloc[0] if not runs.empty and "task" in runs.columns else None
    symbol = runs["symbol"].iloc[0] if not runs.empty and "symbol" in runs.columns else None
    if task is not None and symbol is not None:
        print(f"Task: {task} | Symbol: {symbol} | Runs: {len(runs)}")
    print(
        "DIAGNOSTIC ONLY — never used to select configs or tune the gate. Pre-register "
        "base config + perturbations; fragility flags are findings, not tuning targets."
    )
    print()
    if table.empty:
        print("no usable runs — nothing to summarise")
    else:
        table = table.sort_values(["dimension", "metric"]).reset_index(drop=True)
        show = table[["dimension", "metric", "base", "min", "max", "spread", "flags", "verdict"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nOVERALL: {overall}")
    print("=" * 80 + "\n")

    out: dict = {
        "task": task,
        "symbol": symbol,
        "summary": table.to_dict(orient="records") if not table.empty else [],
        "overall": overall,
    }
    if as_json:
        print(json.dumps(out, indent=2, default=str))
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the research sweep on live data."""
    con = connect()
    try:
        df = run_research(con)
        if not df.empty:
            out_path = "data/research_sweep.csv"
            df.to_csv(out_path, index=False)
            print(f"Results saved to {out_path}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
