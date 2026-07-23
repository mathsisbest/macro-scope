"""Unit tests for Phase 1 canonical splitters and evaluation metrics modules."""

import numpy as np
import pandas as pd
import pytest

from mmi.ml.metrics import (
    ForecastEvaluationResult,
    compute_directional_accuracy,
    compute_ic,
    compute_r2,
    compute_sharpe,
)
from mmi.ml.splitters import feasible_date_range, walk_forward_split


def test_walk_forward_split_standard():
    total_len = 100
    train_size = 50
    test_size = 10

    splits = list(walk_forward_split(total_len, train_size, test_size))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(train_idx) == 50
        assert len(test_idx) == 10
        assert train_idx[-1] + 1 == test_idx[0]


def test_walk_forward_split_single():
    total_len = 100
    train_size = 50
    test_size = 10

    splits = list(walk_forward_split(total_len, train_size, test_size, single_split=True))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert train_idx == list(range(0, 50))
    assert test_idx == list(range(50, 100))


def test_walk_forward_split_expanding():
    total_len = 80
    train_size = 40
    test_size = 20

    splits = list(walk_forward_split(total_len, train_size, test_size, use_all_train=True))
    assert len(splits) == 2
    # First fold: train 0..40, test 40..60
    assert splits[0][0] == list(range(0, 40))
    # Second fold: train 0..60 (expanding!), test 60..80
    assert splits[1][0] == list(range(0, 60))


def test_feasible_date_range():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates})

    first, last = feasible_date_range(df, train_size=5)
    assert first == dates[5]
    assert last == dates[9]

    # Insufficient length
    first_nat, last_nat = feasible_date_range(df, train_size=15)
    assert pd.isna(first_nat) and pd.isna(last_nat)


def test_compute_ic():
    yt = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
    yp = pd.Series([0.01, 0.015, -0.005, 0.025, -0.01])

    ic_val, ic_pval = compute_ic(yt, yp)
    assert ic_val > 0.95
    assert ic_pval < 0.05

    # Zero std edge case
    ic_zero, p_zero = compute_ic(pd.Series([0, 0, 0]), pd.Series([1, 2, 3]))
    assert ic_zero == 0.0
    assert p_zero == 1.0


def test_compute_directional_accuracy():
    yt = pd.Series([0.01, 0.02, -0.01, -0.03, 0.02])
    yp = pd.Series([0.02, 0.01, -0.02, 0.01, 0.01])  # 4 right, 1 wrong (-0.03 vs 0.01)

    metrics = compute_directional_accuracy(yt, yp)
    assert pytest.approx(metrics["direction_accuracy"], 0.01) == 0.8
    assert pytest.approx(metrics["positive_target_rate"], 0.01) == 0.6
    assert pytest.approx(metrics["baseline_direction_accuracy"], 0.01) == 0.6
    assert pytest.approx(metrics["direction_edge"], 0.01) == 0.2


def test_compute_sharpe():
    yt = pd.Series([0.01, 0.02, -0.01, -0.03, 0.02])
    yp = pd.Series([0.02, 0.01, -0.02, -0.01, 0.01])

    sharpe = compute_sharpe(yt, yp, target_horizon=1)
    assert not np.isnan(sharpe)
    assert sharpe > 0.0


def test_compute_r2():
    yt = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
    yp = pd.Series([0.01, 0.015, -0.005, 0.025, -0.01])

    r2_ic = compute_r2(yt, yp, method="ic_signed_sq")
    assert r2_ic > 0.9

    r2_reg = compute_r2(yt, yp, method="regression")
    assert r2_reg > 0.0


def test_forecast_evaluation_result_schema():
    res = ForecastEvaluationResult(ic=0.15, r2=0.0225, model="gb")
    assert res.ic == 0.15
    assert res["ic"] == 0.15
    assert res.get("r2") == 0.0225
    assert res.get("missing_key", "default") == "default"

    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["ic"] == 0.15
    assert d["r2"] == 0.0225


def test_train_latest_forecast_tuning():
    from mmi.ml.forecast import train_latest_forecast

    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(100, 200, 300),
            "high": np.linspace(101, 201, 300),
            "low": np.linspace(99, 199, 300),
            "close": np.linspace(100, 200, 300),
            "daily_return": np.random.randn(300) * 0.01,
            "ret": np.random.randn(300) * 0.01,
        }
    )

    result = train_latest_forecast(
        df=df,
        train_size=100,
        model="gb",
        tune_hyperparameters=True,
    )
    assert "prediction" in result
    assert result["as_of"] == dates[-1]


def test_return_forecast_skill_verdict():
    from mmi.ml.skill_gate import return_forecast_skill_verdict

    df = pd.DataFrame(
        [
            {"model": "return_gb", "symbol": "SPY", "metric": "r2", "value": 0.05},
            {
                "model": "return_gb",
                "symbol": "SPY",
                "metric": "direction_accuracy",
                "value": 0.58,
            },
            {
                "model": "return_gb",
                "symbol": "SPY",
                "metric": "prediction_count",
                "value": 150,
            },
        ]
    )
    verdict = return_forecast_skill_verdict(df, symbol="SPY", model="return_gb")
    assert verdict["cleared"] is True
    assert verdict["oos_r2"] == 0.05

    # Failing case: negative R2
    df_fail = pd.DataFrame(
        [
            {"model": "return_gb", "symbol": "SPY", "metric": "r2", "value": -0.01},
            {
                "model": "return_gb",
                "symbol": "SPY",
                "metric": "direction_accuracy",
                "value": 0.58,
            },
        ]
    )
    verdict_fail = return_forecast_skill_verdict(df_fail, symbol="SPY", model="return_gb")
    assert verdict_fail["cleared"] is False


def test_transform_fallback_marts(tmp_path):
    import duckdb

    from mmi.transform_fallback import build_marts

    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    con.execute("CREATE SCHEMA raw;")
    ddl = """
        CREATE TABLE raw.asset_prices (
            symbol VARCHAR, asset_class VARCHAR, date VARCHAR,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, source VARCHAR
        );
        CREATE TABLE raw.macro_series (
            series_id VARCHAR, date VARCHAR, value DOUBLE,
            source VARCHAR, loaded_at TIMESTAMP
        );
        CREATE TABLE raw.portfolio_returns (
            window_id VARCHAR, strategy VARCHAR, date VARCHAR,
            daily_return DOUBLE, cumulative_return DOUBLE
        );
        CREATE TABLE raw.portfolio_strategy_stats (
            window_id VARCHAR, strategy VARCHAR, ann_return DOUBLE
        );
        CREATE TABLE raw.portfolio_strategy_pairs (
            window_id VARCHAR, strategy_a VARCHAR, strategy_b VARCHAR, corr DOUBLE
        );
        CREATE TABLE raw.portfolio_attribution (
            window_id VARCHAR, strategy VARCHAR, factor VARCHAR, beta DOUBLE
        );
        CREATE TABLE raw.portfolio_btc_effect (
            window_id VARCHAR, btc_alloc DOUBLE, sharpe DOUBLE
        );
        CREATE TABLE raw.portfolio_ml_gate (
            window_id VARCHAR, model VARCHAR, cleared BOOLEAN
        );
    """
    for stmt in ddl.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt + ";")

    build_marts(con)

    tables = [r[0] for r in con.execute("SHOW TABLES FROM marts;").fetchall()]
    assert "fct_portfolio_regime_performance" in tables
    assert "fct_recession_risk" in tables
    assert "fct_market_macro" in tables


def test_vol_rich_plus_core_assets_features():
    from mmi.ml.features import feature_columns, make_features

    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df_spy = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(300, 400, 100),
            "high": np.linspace(305, 405, 100),
            "low": np.linspace(295, 395, 100),
            "close": np.linspace(300, 400, 100),
            "daily_return": np.random.randn(100) * 0.01,
            "ret": np.random.randn(100) * 0.01,
        }
    )
    df_tlt = pd.DataFrame({"date": dates, "daily_return": np.random.randn(100) * 0.01})
    df_gld = pd.DataFrame({"date": dates, "daily_return": np.random.randn(100) * 0.01})
    df_btc = pd.DataFrame({"date": dates, "daily_return": np.random.randn(100) * 0.02})

    asset_dfs = {"TLT": df_tlt, "GLD": df_gld, "BTC": df_btc}
    feats = make_features(df_spy, feature_set="vol_rich_plus", asset_dfs=asset_dfs)

    cols = feature_columns("vol_rich_plus")
    assert "spy_tlt_spread_20d" in cols
    assert "spy_gld_spread_20d" in cols
    assert "btc_spy_spread_20d" in cols
    assert "spy_tlt_spread_20d" in feats.columns
    assert "btc_spy_spread_20d" in feats.columns
