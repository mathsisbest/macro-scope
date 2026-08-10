"""Tests for the robustness / sensitivity analysis in mmi.ml.research.

Covers the pure parts (metric aggregation, fragility classification, probe
mapping, gate helpers) plus smoke tests for the run driver on both tasks.
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from mmi.ml.metrics import ForecastEvaluationResult
from mmi.ml.research import (
    Perturbation,
    _base_knob_value,
    _perturbation_overrides,
    _return_gate_cleared,
    _vol_gate_cleared,
    aggregate_robustness,
    classify_robustness,
    default_perturbations,
    run_robustness_analysis,
    summarize_robustness,
)


@pytest.fixture
def mock_con():
    """In-memory DuckDB with a minimal marts schema and 160 SPY rows.

    160 rows (not 100) so vol probes with ``horizon=7`` keep >= 60 trainable
    dev rows and every probe actually runs.
    """
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA marts")
    con.execute("""
        CREATE TABLE marts.fct_asset_daily (
            symbol VARCHAR,
            date TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            daily_return DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE marts.fct_macro_indicator (
            date TIMESTAMP,
            series_id VARCHAR,
            value DOUBLE
        )
    """)

    dates = pd.bdate_range("2023-01-01", periods=160)
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, size=160)
    rows = []
    for d, r in zip(dates, rets, strict=True):
        rows.append(("SPY", d, 100.0, 101.0, 99.0, 100.0 + r, r))
    con.executemany(
        "INSERT INTO marts.fct_asset_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    yield con
    con.close()


def _runs_df() -> pd.DataFrame:
    """A synthetic robustness runs frame: base + 2 probes × 2 metrics."""
    return pd.DataFrame(
        [
            # base run (dimension="base", value=None)
            {
                "dimension": "base",
                "value": None,
                "metric": "ic",
                "metric_value": 0.05,
                "gate_cleared": True,
            },
            {
                "dimension": "base",
                "value": None,
                "metric": "oos_r2",
                "metric_value": 0.20,
                "gate_cleared": True,
            },
            # horizon=1 probe
            {
                "dimension": "horizon",
                "value": 1,
                "metric": "ic",
                "metric_value": 0.04,
                "gate_cleared": True,
            },
            {
                "dimension": "horizon",
                "value": 1,
                "metric": "oos_r2",
                "metric_value": 0.15,
                "gate_cleared": True,
            },
            # horizon=5 probe
            {
                "dimension": "horizon",
                "value": 5,
                "metric": "ic",
                "metric_value": 0.02,
                "gate_cleared": True,
            },
            {
                "dimension": "horizon",
                "value": 5,
                "metric": "oos_r2",
                "metric_value": 0.05,
                "gate_cleared": True,
            },
        ]
    )


def _calm_runs_df() -> pd.DataFrame:
    """A synthetic robustness runs frame with small swings (all robust)."""
    return pd.DataFrame(
        [
            {
                "dimension": "base",
                "value": None,
                "metric": "ic",
                "metric_value": 0.05,
                "gate_cleared": True,
            },
            {
                "dimension": "base",
                "value": None,
                "metric": "oos_r2",
                "metric_value": 0.20,
                "gate_cleared": True,
            },
            {
                "dimension": "horizon",
                "value": 1,
                "metric": "ic",
                "metric_value": 0.04,
                "gate_cleared": True,
            },
            {
                "dimension": "horizon",
                "value": 1,
                "metric": "oos_r2",
                "metric_value": 0.19,
                "gate_cleared": True,
            },
            {
                "dimension": "horizon",
                "value": 5,
                "metric": "ic",
                "metric_value": 0.06,
                "gate_cleared": True,
            },
            {
                "dimension": "horizon",
                "value": 5,
                "metric": "oos_r2",
                "metric_value": 0.21,
                "gate_cleared": True,
            },
        ]
    )


# ---------------------------------------------------------------------------
# default_perturbations / probe mapping
# ---------------------------------------------------------------------------


def test_default_perturbations_vol():
    probes = default_perturbations(
        {"model_name": "rv_har", "horizon": 5, "min_dev": 60, "n_splits": 5}
    )
    by_dim = {p.dimension: p.value_options for p in probes}
    assert by_dim["horizon"] == (3, 7)
    assert by_dim["min_periods"] == (40, 80)
    assert by_dim["window"] == (3, 7)
    assert "seed" not in by_dim  # OLS models are deterministic


def test_default_perturbations_vol_gb_gets_seed():
    probes = default_perturbations({"model_name": "rv_gb"})
    by_dim = {p.dimension: p.value_options for p in probes}
    assert by_dim["seed"] == (1, 2)


def test_default_perturbations_vol_floors_and_dedupes():
    probes = default_perturbations(
        {"model_name": "rv_har", "horizon": 1, "min_dev": 10, "n_splits": 2}
    )
    by_dim = {p.dimension: p.value_options for p in probes}
    assert by_dim["horizon"] == (3,)  # max(1, -1) = 1 == base -> dropped
    assert by_dim["min_periods"] == (1, 30)  # max(1, -10) = 1, base is 10 -> kept
    assert by_dim["window"] == (4,)  # max(2, 0) = 2 == base -> dropped


def test_default_perturbations_return():
    probes = default_perturbations({}, task="return")
    by_dim = {p.dimension: p.value_options for p in probes}
    assert by_dim["train_size"] == (150, 350)
    assert by_dim["horizon"] == (2,)  # 1-1 = 0 floored to 1 == base -> dropped
    assert by_dim["window"] == (True,)
    assert by_dim["seed"] == (41, 43)


def test_default_perturbations_unknown_task_raises():
    with pytest.raises(ValueError, match="unknown task"):
        default_perturbations({}, task="bogus")


def test_perturbation_overrides_vol():
    cfg = {"model_name": "rv_gb"}
    assert _perturbation_overrides(cfg, "vol", "horizon", 7) == {"horizon": 7}
    assert _perturbation_overrides(cfg, "vol", "min_periods", 80) == {"min_dev": 80}
    assert _perturbation_overrides(cfg, "vol", "window", 3) == {"n_splits": 3}
    assert _perturbation_overrides(cfg, "vol", "seed", 1) == {"model_params": {"random_state": 1}}


def test_perturbation_overrides_seed_rejected_for_deterministic_models():
    with pytest.raises(ValueError, match="only meaningful for model 'rv_gb'"):
        _perturbation_overrides({"model_name": "rv_har"}, "vol", "seed", 1)


def test_perturbation_overrides_return():
    assert _perturbation_overrides({}, "return", "train_size", 150) == {"train_size": 150}
    assert _perturbation_overrides({}, "return", "horizon", 5) == {"target_horizon": 5}
    assert _perturbation_overrides({}, "return", "window", True) == {"use_all_train": True}
    assert _perturbation_overrides({}, "return", "seed", 41) == {
        "model_kwargs": {"random_state": 41}
    }


def test_perturbation_overrides_unknown_dimension_raises():
    with pytest.raises(ValueError, match="unknown perturbation dimension"):
        _perturbation_overrides({}, "vol", "bogus", 1)


def test_base_knob_value():
    vol_cfg = {"model_name": "rv_har", "horizon": 5, "min_dev": 60, "n_splits": 5}
    assert _base_knob_value(vol_cfg, "vol", "horizon") == 5
    assert _base_knob_value(vol_cfg, "vol", "min_periods") == 60
    assert _base_knob_value(vol_cfg, "vol", "window") == 5
    assert _base_knob_value(vol_cfg, "vol", "seed") == 0
    ret_cfg = {"train_size": 250, "target_horizon": 1, "use_all_train": False}
    assert _base_knob_value(ret_cfg, "return", "train_size") == 250
    assert _base_knob_value(ret_cfg, "return", "horizon") == 1
    assert _base_knob_value(ret_cfg, "return", "window") is False
    with pytest.raises(ValueError, match="unknown perturbation dimension"):
        _base_knob_value(vol_cfg, "vol", "bogus")


# ---------------------------------------------------------------------------
# aggregate_robustness (pure)
# ---------------------------------------------------------------------------


def test_aggregate_robustness():
    agg = aggregate_robustness(_runs_df())
    horizon = agg[agg["dimension"] == "horizon"].set_index("metric")
    assert horizon.loc["ic", "base"] == pytest.approx(0.05)
    assert horizon.loc["ic", "min"] == pytest.approx(0.02)
    assert horizon.loc["ic", "max"] == pytest.approx(0.04)
    assert horizon.loc["ic", "n_probes"] == 2
    assert horizon.loc["ic", "spread"] == pytest.approx(0.03)  # |0.02 - 0.05|
    assert horizon.loc["oos_r2", "spread"] == pytest.approx(0.15)  # |0.05 - 0.20|


def test_aggregate_robustness_missing_base():
    runs = _runs_df()
    runs = runs[runs["dimension"] != "base"].copy()
    agg = aggregate_robustness(runs)
    assert len(agg) == 2
    assert np.isnan(agg.iloc[0]["base"])
    assert np.isnan(agg.iloc[0]["spread"])


def test_aggregate_robustness_drops_nan_probes():
    runs = _runs_df().copy()
    runs.loc[2, "metric_value"] = np.nan  # one NaN probe row
    agg = aggregate_robustness(runs)
    horizon_ic = agg[(agg["dimension"] == "horizon") & (agg["metric"] == "ic")].iloc[0]
    assert horizon_ic["n_probes"] == 1
    assert horizon_ic["min"] == pytest.approx(0.02)


def test_aggregate_robustness_empty():
    assert aggregate_robustness(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# classify_robustness (pure)
# ---------------------------------------------------------------------------


def test_classify_robustness_robust():
    verdicts = classify_robustness(_calm_runs_df())
    horizon = verdicts[verdicts["dimension"] == "horizon"].set_index("metric")
    assert horizon.loc["ic", "verdict"] == "robust"  # no ref cross, swings <= 0.05
    assert horizon.loc["oos_r2", "verdict"] == "robust"


def test_classify_robustness_sign_flip():
    runs = _calm_runs_df().copy()
    runs.loc[4, "metric_value"] = -0.01  # ic probe crosses the 0.0 no-skill reference
    verdicts = classify_robustness(runs)
    ic_row = verdicts[(verdicts["dimension"] == "horizon") & (verdicts["metric"] == "ic")].iloc[0]
    assert "sign_flip" in ic_row["flags"]
    assert ic_row["verdict"] == "fragile"
    # oos_r2 unaffected (still small swings, no ref cross)
    r2_row = verdicts[(verdicts["dimension"] == "horizon") & (verdicts["metric"] == "oos_r2")].iloc[
        0
    ]
    assert r2_row["verdict"] == "robust"


def test_classify_robustness_direction_accuracy_reference_is_half():
    runs = pd.DataFrame(
        [
            {
                "dimension": "base",
                "value": None,
                "metric": "direction_accuracy",
                "metric_value": 0.51,
                "gate_cleared": True,
            },
            {
                "dimension": "horizon",
                "value": 1,
                "metric": "direction_accuracy",
                "metric_value": 0.48,
                "gate_cleared": True,
            },
        ]
    )
    verdicts = classify_robustness(runs)
    row = verdicts.iloc[0]
    assert "sign_flip" in row["flags"]  # crossed the 0.5 coin-flip reference
    assert row["verdict"] == "fragile"


def test_classify_robustness_large_swing():
    runs = pd.DataFrame(
        [
            {
                "dimension": "base",
                "value": None,
                "metric": "oos_r2",
                "metric_value": 0.50,
                "gate_cleared": True,
            },
            {
                "dimension": "horizon",
                "value": 1,
                "metric": "oos_r2",
                "metric_value": 0.40,
                "gate_cleared": True,
            },
        ]
    )
    verdicts = classify_robustness(runs)
    assert "large_swing" in verdicts.iloc[0]["flags"]
    assert verdicts.iloc[0]["verdict"] == "fragile"


def test_classify_robustness_gate_flip():
    runs = pd.DataFrame(
        [
            {
                "dimension": "base",
                "value": None,
                "metric": "oos_r2",
                "metric_value": 0.30,
                "gate_cleared": True,
            },
            {
                "dimension": "horizon",
                "value": 1,
                "metric": "oos_r2",
                "metric_value": 0.28,
                "gate_cleared": False,
            },
        ]
    )
    verdicts = classify_robustness(runs)
    row = verdicts.iloc[0]
    assert "gate_flip" in row["flags"]
    assert row["verdict"] == "fragile"  # gate clears<->fails is fragile even without big swings


def test_classify_robustness_inconclusive_no_base():
    runs = _runs_df()
    runs = runs[runs["dimension"] != "base"].copy()
    verdicts = classify_robustness(runs)
    assert (verdicts["verdict"] == "inconclusive").all()
    assert all("no_base" in f for f in verdicts["flags"])


def test_classify_robustness_inconclusive_no_probes():
    runs = _runs_df().copy()
    probe_mask = (runs["metric"] == "oos_r2") & (runs["dimension"] != "base")
    runs.loc[probe_mask, "metric_value"] = np.nan  # oos_r2 probes all invalid
    verdicts = classify_robustness(runs)
    r2_row = verdicts[verdicts["metric"] == "oos_r2"].iloc[0]
    assert "no_probes" in r2_row["flags"]
    assert r2_row["verdict"] == "inconclusive"
    ic_row = verdicts[verdicts["metric"] == "ic"].iloc[0]
    assert ic_row["verdict"] == "robust"


def test_classify_robustness_min_probes_required():
    runs = _calm_runs_df().copy()
    runs = runs[runs["value"] != 5]  # only 1 probe left per metric
    verdicts = classify_robustness(runs, min_probes=2)
    assert (verdicts["verdict"] == "inconclusive").all()


def test_classify_robustness_empty():
    assert classify_robustness(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# gate helpers (pure)
# ---------------------------------------------------------------------------


def _vol_metrics(**overrides) -> dict:
    metrics = {
        "oos_r2": 0.30,
        "qlike_skill_ratio": 0.80,
        "folds_passed": 4,
        "n_folds": 5,
        "n_obs": 300,
    }
    metrics.update(overrides)
    return metrics


def test_vol_gate_cleared():
    assert _vol_gate_cleared(_vol_metrics(), "rv_har", "SPY") is True
    assert _vol_gate_cleared(_vol_metrics(oos_r2=0.05), "rv_har", "SPY") is False
    assert _vol_gate_cleared(_vol_metrics(n_obs=100), "rv_har", "SPY") is False


def test_vol_gate_cleared_not_applicable():
    assert _vol_gate_cleared(_vol_metrics(), "rv_ridge", "SPY") is None  # not the gated model
    assert _vol_gate_cleared({}, "rv_har", "SPY") is None  # skipped run


def _return_result(**overrides) -> ForecastEvaluationResult:
    fields = {
        "ic": 0.05,
        "direction_accuracy": 0.55,
        "r2": 0.02,
        "folds_passed": 4,
        "n_folds": 5,
        "prediction_count": 300,
        "annualised_alpha": 0.02,
        "turnover_adjusted_sharpe": 1.0,
        "model": "gb",
    }
    fields.update(overrides)
    return ForecastEvaluationResult(**fields)


def test_return_gate_cleared():
    assert _return_gate_cleared(_return_result(), "gb", "SPY") is True
    assert _return_gate_cleared(_return_result(direction_accuracy=0.49), "gb", "SPY") is False
    assert _return_gate_cleared(_return_result(r2=-0.01), "gb", "SPY") is False


def test_return_gate_cleared_skipped_run_is_none():
    assert _return_gate_cleared(_return_result(prediction_count=0), "gb", "SPY") is None


# ---------------------------------------------------------------------------
# run_robustness_analysis driver (smoke)
# ---------------------------------------------------------------------------


def test_run_robustness_analysis_vol_smoke(mock_con):
    runs = run_robustness_analysis(
        mock_con,
        symbol="SPY",
        task="vol",
        base_config={"model_name": "rv_har", "feature_set": "vol"},
        perturbations=[Perturbation("horizon", (3, 7))],
    )
    assert not runs.empty
    assert set(runs.columns) == {
        "task",
        "symbol",
        "dimension",
        "value",
        "metric",
        "metric_value",
        "gate_cleared",
    }
    assert runs["task"].unique().tolist() == ["vol"]
    assert set(runs["dimension"].unique()) == {"base", "horizon"}
    assert len(runs) == 3 * (1 + 2)  # 3 metrics × (base + 2 probes)
    assert set(runs["metric"].unique()) == {"oos_r2", "qlike_skill_ratio", "folds_passed"}
    assert runs["metric_value"].notna().all()


def test_run_robustness_analysis_return_smoke(mock_con):
    runs = run_robustness_analysis(
        mock_con,
        symbol="SPY",
        task="return",
        base_config={"model": "gb", "feature_set": "default", "train_size": 60, "test_size": 10},
        perturbations=[Perturbation("train_size", (50, 70))],
    )
    assert not runs.empty
    assert runs["task"].unique().tolist() == ["return"]
    assert set(runs["metric"].unique()) == {"ic", "direction_accuracy", "r2"}
    assert len(runs) == 3 * (1 + 2)
    assert runs["metric_value"].notna().all()


def test_run_robustness_analysis_default_perturbations(mock_con):
    runs = run_robustness_analysis(
        mock_con,
        symbol="SPY",
        task="vol",
        base_config={"model_name": "rv_har"},
    )
    assert set(runs["dimension"].unique()) == {"base", "horizon", "min_periods", "window"}
    # base + (2+2+2) probes × 3 metrics
    assert len(runs) == 3 * (1 + 6)


def test_run_robustness_analysis_skips_probe_equal_to_base(mock_con):
    runs = run_robustness_analysis(
        mock_con,
        task="vol",
        perturbations=[Perturbation("horizon", (5, 7))],  # 5 == base horizon
    )
    assert set(runs["dimension"].unique()) == {"base", "horizon"}
    assert len(runs[runs["dimension"] == "horizon"]) == 3  # only the 7 probe ran


def test_run_robustness_analysis_unknown_task_raises():
    with pytest.raises(ValueError, match="unknown task"):
        run_robustness_analysis(None, task="bogus")


def test_run_robustness_analysis_unknown_dimension_raises(mock_con):
    with pytest.raises(ValueError, match="unknown perturbation dimension"):
        run_robustness_analysis(mock_con, task="vol", perturbations=[Perturbation("bogus", (1,))])


# ---------------------------------------------------------------------------
# summarize_robustness
# ---------------------------------------------------------------------------


def test_summarize_robustness_smoke(capsys):
    out = summarize_robustness(_calm_runs_df())
    captured = capsys.readouterr()
    assert "ROBUSTNESS SENSITIVITY ANALYSIS" in captured.out
    assert "DIAGNOSTIC ONLY" in captured.out
    assert out["overall"] == "robust"
    assert len(out["summary"]) == 2
    assert out["summary"][0]["verdict"] == "robust"


def test_summarize_robustness_as_json(capsys):
    out = summarize_robustness(_calm_runs_df(), as_json=True)
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("{"))
    payload = json.loads("\n".join(lines[start:]))
    assert payload["overall"] == out["overall"]
    assert payload["task"] is None


def test_summarize_robustness_fragile_overall():
    runs = _runs_df().copy()
    runs.loc[4, "metric_value"] = -0.01  # sign flip on ic
    out = summarize_robustness(runs)
    assert out["overall"] == "fragile"
