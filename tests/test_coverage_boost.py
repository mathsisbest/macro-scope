"""Coverage tests for target modules: compute, forecast, features, backtest, forecast_panel."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mmi.ml.features import feature_columns, make_features
from mmi.ml.forecast import evaluate_forecast
from mmi.ml.forecast_panel import ForecastBacktest, _compute_panel_metrics, _empty_panel_result
from mmi.portfolio.backtest import rebalance_dates, run_backtest, run_backtest_full
from mmi.portfolio.compute import (
    btc_aligned_returns,
    compute_all_predictions,
    compute_ml_mu_panel,
    compute_portfolio_returns,
    window_asset_daily,
)

# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════


def _ohlc_df(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
    high = close * (1 + rng.uniform(0.001, 0.02, n_days))
    low = close * (1 - rng.uniform(0.001, 0.02, n_days))
    open_ = low + rng.uniform(0, 1, n_days) * (high - low)
    daily_return = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "daily_return": daily_return,
        }
    )


def _asset_daily(
    n_days: int = 100, symbols: tuple[str, ...] = ("SPY", "TLT"), seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for sym in symbols:
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "asset_class": "stock",
                    "daily_return": float(rng.normal(0.0005, 0.01)),
                }
            )
    btcp = 100 * np.cumprod(1 + rng.normal(0, 0.02, n_days))
    btc_ret = np.diff(btcp, prepend=btcp[0]) / btcp
    for i, d in enumerate(dates):
        rows.append(
            {
                "date": d,
                "symbol": "BTC",
                "asset_class": "crypto",
                "daily_return": float(btc_ret[i]),
            }
        )
    return pd.DataFrame(rows)


def _returns_panel(n_days: int = 400, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    data = rng.normal(0.0004, 0.01, size=(n_days, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _macro_df(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "date": dates,
            "T10Y2Y": rng.normal(0.5, 0.5, n_days),
            "DGS10": rng.normal(4.0, 0.5, n_days),
            "DGS2": rng.normal(3.5, 0.5, n_days),
            "DGS3MO": rng.normal(3.0, 1.0, n_days),
            "FEDFUNDS": rng.normal(3.0, 1.0, n_days),
            "VIXCLS": rng.normal(20, 5, n_days),
            "DCOILWTICO": 70 + rng.normal(0, 2, n_days),
            "DTWEXBGS": 120 + rng.normal(0, 1, n_days),
            "ICSA": 200 + rng.normal(0, 10, n_days),
            "NFCI": rng.normal(-0.5, 0.2, n_days),
            "INDPRO": 100 + rng.normal(0, 1, n_days).cumsum(),
            "CPIAUCSL": 250 + rng.normal(0, 0.5, n_days).cumsum(),
            "PCEPILFE": 200 + rng.normal(0, 0.3, n_days).cumsum(),
            "UNRATE": rng.normal(4.0, 0.5, n_days),
            "PAYEMS": 150000 + rng.normal(0, 100, n_days).cumsum(),
            "M2SL": 20000 + rng.normal(0, 50, n_days).cumsum(),
            "WALCL": 8000 + rng.normal(0, 10, n_days).cumsum(),
            "UMCSENT": rng.normal(70, 5, n_days),
            "SAHMREALTIME": rng.normal(0.1, 0.05, n_days),
        }
    )


def _asset_dfs(n_days: int = 120, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    dfs = {}
    for sym in ("GLD", "TLT", "VEA", "TIP", "BTC"):
        dfs[sym] = pd.DataFrame(
            {
                "date": dates,
                "daily_return": rng.normal(0.0003, 0.008, n_days),
            }
        )
    return dfs


# ══════════════════════════════════════════════════════════════════
# compute.py
# ══════════════════════════════════════════════════════════════════


class TestComputePortfolioReturns:
    def test_no_ml_mu_panel(self):
        result = compute_portfolio_returns(_asset_daily(50, ("SPY", "TLT")))
        assert "strategy" in result.columns
        assert result["strategy"].iloc[0] == "equal_weight"
        assert "cumulative_return" in result.columns

    def test_empty_ml_mu_panel(self):
        empty = pd.DataFrame(columns=["date", "symbol", "mu"])
        result = compute_portfolio_returns(_asset_daily(50, ("SPY", "TLT")), ml_mu_panel=empty)
        assert result["strategy"].iloc[0] == "equal_weight"

    def test_ml_mu_no_overlap(self):
        df = _asset_daily(50, ("SPY", "TLT"))
        mu_panel = pd.DataFrame(
            {"date": pd.date_range("1980-01-01", periods=5), "symbol": "SPY", "mu": 0.001}
        )
        result = compute_portfolio_returns(df, ml_mu_panel=mu_panel, window="test")
        assert "equal_weight" in result["strategy"].unique()

    def test_with_ml_mu_panel(self):
        df = _asset_daily(120, ("SPY", "TLT"))
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2020-01-01", periods=100)
        mu_rows = []
        for d in dates:
            for s in ["SPY", "TLT"]:
                mu_rows.append({"date": d, "symbol": s, "mu": float(rng.uniform(-0.01, 0.02))})
        mu_panel = pd.DataFrame(mu_rows)
        result = compute_portfolio_returns(df, ml_mu_panel=mu_panel, window="test")
        strategies = result["strategy"].unique()
        assert "equal_weight" in strategies
        assert "ml_tilt" in strategies
        assert "ml_regime" in strategies


class TestBtcAlignedReturns:
    def test_with_btc(self):
        result = btc_aligned_returns(_asset_daily(80, ("SPY", "TLT")))
        assert "date" in result.columns
        assert "daily_return" in result.columns
        assert not result.empty

    def test_without_btc(self):
        df = _asset_daily(50, ("SPY", "TLT"))
        df = df[df["symbol"] != "BTC"].reset_index(drop=True)
        result = btc_aligned_returns(df)
        assert result.empty


class TestWindowAssetDaily:
    def test_non_crypto_filter(self):
        df = _asset_daily(50, ("SPY", "TLT", "GLD"))
        result = window_asset_daily(df, window_id="ex_btc_2002")
        assert "symbol" in result.columns
        assert result["symbol"].nunique() > 0

    def test_ex_btc_2002_empty_non_crypto(self):
        df = _asset_daily(50, ("GLD",))
        df = df[df["symbol"] != "BTC"].reset_index(drop=True)
        result = window_asset_daily(df, window_id="ex_btc_2002")
        assert result.empty or result["symbol"].nunique() > 0

    def test_ex_btc_2015(self):
        df = _asset_daily(50, ("SPY", "TLT", "GLD"))
        floor = pd.Timestamp("2020-01-10")
        result = window_asset_daily(df, window_id="ex_btc_2015", btc_floor=floor)
        assert not result.empty

    def test_ex_btc_2015_no_btc_floor(self):
        df = _asset_daily(50, ("SPY", "TLT", "GLD"))
        result = window_asset_daily(df, window_id="ex_btc_2015", btc_floor=None)
        assert not result.empty

    def test_inc_btc_2015(self):
        df = _asset_daily(80, ("SPY", "TLT", "GLD"))
        btc_a = btc_aligned_returns(df)
        floor = pd.Timestamp("2020-01-10")
        result = window_asset_daily(
            df, window_id="inc_btc_2015", btc_floor=floor, btc_aligned=btc_a
        )
        assert not result.empty

    def test_inc_btc_2015_no_btc_aligned(self):
        df = _asset_daily(80, ("SPY", "TLT", "GLD"))
        result = window_asset_daily(
            df, window_id="inc_btc_2015", btc_floor=pd.Timestamp("2020-01-10")
        )
        assert not result.empty

    def test_unknown_window(self):
        df = _asset_daily(50, ("SPY", "TLT", "GLD"))
        result = window_asset_daily(df, window_id="unknown")
        assert not result.empty


class TestComputeMlMuPanel:
    def test_basic_default_features(self):
        df = _asset_daily(200, ("SPY",))
        mu_panel, gate = compute_ml_mu_panel(
            df,
            train_size=50,
            test_size=10,
            target_horizon=21,
            feature_set="default",
        )
        assert "mu" in mu_panel.columns
        assert "forecast_skill" in gate.columns

    def test_too_few_rows_skips_asset(self):
        df = _asset_daily(30, ("SPY", "TLT"))
        mu_panel, _ = compute_ml_mu_panel(
            df, train_size=50, test_size=10, target_horizon=21, feature_set="default"
        )
        assert mu_panel.empty

    def test_missing_ohlc_for_vol_fset_skips_asset(self):
        df = _asset_daily(200, ("SPY",))
        df = df[["date", "symbol", "daily_return"]]
        mu_panel, _ = compute_ml_mu_panel(
            df, train_size=50, test_size=10, target_horizon=21, feature_set="vol"
        )
        assert mu_panel.empty

    def test_with_asset_daily_full(self):
        df = _asset_daily(200, ("SPY",))
        full = _asset_daily(250, ("SPY", "TLT"))
        mu_panel, _ = compute_ml_mu_panel(
            df,
            train_size=50,
            test_size=10,
            target_horizon=21,
            feature_set="default",
            asset_daily_full=full,
        )
        assert "mu" in mu_panel.columns


class TestComputeAllPredictions:
    def test_basic(self):
        db = mock.MagicMock()
        df = _ohlc_df(250)
        db.prices_df.return_value = df
        db.macro_df = _macro_df(250)
        db.asset_dfs = _asset_dfs(250)
        result = compute_all_predictions(
            db,
            universe=["SPY"],
            train_size=50,
            test_size=15,
            single_split=True,
            feature_set="default",
            target_horizon=1,
        )
        assert "date" in result.columns
        assert "symbol" in result.columns
        assert "pred_ret" in result.columns
        assert len(result) > 0

    def test_default_universe(self):
        db = mock.MagicMock()
        df = _ohlc_df(250)
        db.prices_df.return_value = df
        db.macro_df = None
        db.asset_dfs = None
        result = compute_all_predictions(
            db,
            train_size=50,
            test_size=15,
            single_split=True,
            target_horizon=1,
        )
        assert "pred_ret" in result.columns

    def test_nodata_returns_empty(self):
        db = mock.MagicMock()
        db.prices_df.return_value = _ohlc_df(30)
        db.macro_df = None
        db.asset_dfs = None
        result = compute_all_predictions(
            db,
            universe=["NODATA"],
            train_size=50,
            test_size=15,
            target_horizon=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_walk_forward_no_single_split(self):
        db = mock.MagicMock()
        db.prices_df.return_value = _ohlc_df(500)
        db.macro_df = None
        db.asset_dfs = None
        result = compute_all_predictions(
            db,
            universe=["SPY"],
            train_size=80,
            test_size=20,
            feature_set="default",
            target_horizon=1,
        )
        assert "pred_ret" in result.columns


# ══════════════════════════════════════════════════════════════════
# forecast.py
# ══════════════════════════════════════════════════════════════════


class TestEvaluateForecast:
    def test_single_split_default(self):
        df = _ohlc_df(200)
        result = evaluate_forecast(
            df[["date", "close", "daily_return"]],
            train_size=50,
            test_size=15,
            single_split=True,
            feature_set="default",
        )
        assert result["prediction_count"] > 0
        assert "predictions" in result
        assert "dates" in result
        assert "ic" in result
        assert "ic_pvalue" in result
        assert "r2" in result
        assert result["model"] == "gb"
        assert result["feature_set"] == "default"

    def test_single_split_vol_adjusted(self):
        df = _ohlc_df(200)
        result = evaluate_forecast(
            df[["date", "close", "daily_return"]],
            train_size=50,
            test_size=15,
            single_split=True,
            target_type="vol_adjusted",
        )
        assert result["prediction_count"] > 0

    def test_single_split_excess(self):
        df = _ohlc_df(200)
        result = evaluate_forecast(
            df[["date", "close", "daily_return"]],
            train_size=50,
            test_size=15,
            single_split=True,
            target_type="excess",
        )
        assert result["prediction_count"] >= 0

    def test_walk_forward(self):
        df = _ohlc_df(300)
        result = evaluate_forecast(
            df[["date", "close", "daily_return"]],
            train_size=80,
            test_size=20,
            feature_set="default",
        )
        assert result["prediction_count"] > 0
        assert result["n_models"] > 0

    def test_walk_forward_lgb(self):
        df = _ohlc_df(300)
        result = evaluate_forecast(
            df[["date", "close", "daily_return"]],
            train_size=80,
            test_size=20,
            model="lgb",
            feature_set="default",
        )
        assert result["prediction_count"] > 0

    def test_empty_result_on_too_few_rows(self):
        df = _ohlc_df(30)
        result = evaluate_forecast(
            df[["date", "close", "daily_return"]],
            train_size=50,
            test_size=10,
        )
        assert result["prediction_count"] == 0

    def test_vol_feature_set(self):
        df = _ohlc_df(200)
        result = evaluate_forecast(
            df,
            train_size=50,
            test_size=15,
            single_split=True,
            feature_set="vol",
        )
        assert result["prediction_count"] > 0

    def test_expanding_window(self):
        df = _ohlc_df(300)
        result = evaluate_forecast(
            df[["date", "close", "daily_return"]],
            train_size=80,
            test_size=20,
            use_all_train=True,
            feature_set="default",
        )
        assert result["prediction_count"] > 0


# ══════════════════════════════════════════════════════════════════
# features.py
# ══════════════════════════════════════════════════════════════════


class TestFeatureColumns:
    def test_vol_medium_has_medium_names(self):
        cols = feature_columns("vol_medium")
        assert "ret_momentum_63d" in cols
        assert "vol_of_vol_5d" in cols
        assert "vol_dispersion" in cols

    def test_vol_rich_has_rich_names(self):
        cols = feature_columns("vol_rich")
        assert "ret_kurt_20d" in cols
        assert "corr_spy_tlt_20d" in cols

    def test_vol_rich_plus_has_extended(self):
        cols = feature_columns("vol_rich_plus")
        assert "indpro_growth_1y" in cols
        assert "breakeven_inflation_20d" in cols

    def test_mom_rev(self):
        cols = feature_columns("mom_rev")
        assert "mom_21d" in cols
        assert "rev_5d" in cols


class TestMakeFeaturesExtended:
    def test_vol_macro(self):
        df = _ohlc_df(100)
        feats = make_features(df, feature_set="vol_macro")
        cols = feature_columns("vol_macro")
        for c in cols:
            assert c in feats.columns, f"Missing {c}"

    def test_vol_macro_with_macro_df(self):
        df = _ohlc_df(100)
        macro = _macro_df(100)
        feats = make_features(df, feature_set="vol_macro", macro_df=macro)
        cols = feature_columns("vol_macro")
        for c in cols:
            assert c in feats.columns, f"Missing {c}"

    def test_vol_medium(self):
        df = _ohlc_df(150)
        feats = make_features(df, feature_set="vol_medium")
        cols = feature_columns("vol_medium")
        for c in cols:
            assert c in feats.columns, f"Missing {c}"

    def test_vol_medium_with_macro(self):
        df = _ohlc_df(150)
        macro = _macro_df(150)
        feats = make_features(df, feature_set="vol_medium", macro_df=macro)
        cols = feature_columns("vol_medium")
        for c in cols:
            assert c in feats.columns, f"Missing {c}"

    def test_vol_rich(self):
        df = _ohlc_df(150)
        macro = _macro_df(150)
        adfs = _asset_dfs(150)
        feats = make_features(df, feature_set="vol_rich", macro_df=macro, asset_dfs=adfs)
        cols = feature_columns("vol_rich")
        for c in cols:
            assert c in feats.columns, f"Missing {c}"

    def test_vol_rich_with_asset_dfs(self):
        df = _ohlc_df(150)
        macro = _macro_df(150)
        adfs = _asset_dfs(150)
        feats = make_features(df, feature_set="vol_rich", macro_df=macro, asset_dfs=adfs)
        cols = feature_columns("vol_rich")
        for c in cols:
            assert c in feats.columns, f"Missing {c}"

    def test_vol_rich_plus(self):
        df = _ohlc_df(200)
        macro = _macro_df(200)
        adfs = _asset_dfs(200)
        feats = make_features(df, feature_set="vol_rich_plus", macro_df=macro, asset_dfs=adfs)
        cols = feature_columns("vol_rich_plus")
        for c in cols:
            assert c in feats.columns, f"Missing {c}"

    def test_vol_rich_plus_with_macro_and_asset_dfs(self):
        df = _ohlc_df(200)
        macro = _macro_df(200)
        adfs = _asset_dfs(200)
        feats = make_features(df, feature_set="vol_rich_plus", macro_df=macro, asset_dfs=adfs)
        cols = feature_columns("vol_rich_plus")
        for c in cols:
            assert c in feats.columns, f"Missing {c}"


# ══════════════════════════════════════════════════════════════════
# backtest.py
# ══════════════════════════════════════════════════════════════════


class TestBacktestFull:
    def test_run_backtest_full_contributions(self):
        panel = _returns_panel(400, 3)
        ret, contrib = run_backtest_full(panel, strategy="equal_weight", lookback=60, freq="M")
        assert "daily_return" in ret.columns
        assert "cumulative_return" in ret.columns
        assert len(contrib) > 0

    def test_mvo_histmean(self):
        panel = _returns_panel(400, 3)
        ret = run_backtest(panel, strategy="mvo_histmean", lookback=60, freq="M")
        assert ret["daily_return"].notna().any()

    def test_mvo_ml(self):
        panel = _returns_panel(400, 3)
        mu_rows = []
        for d in panel.index:
            for s in ["A0", "A1", "A2"]:
                mu_rows.append({"date": d, "symbol": s, "mu": 0.001})
        mu_panel = pd.DataFrame(mu_rows)
        ret, contrib = run_backtest_full(
            panel, strategy="mvo_ml", lookback=60, freq="M", mu_panel=mu_panel
        )
        assert "daily_return" in ret.columns

    def test_tsmom_overlay(self):
        panel = _returns_panel(400, 3)
        tsmom_rows = []
        for d in panel.index:
            for s in ["A0", "A1", "A2"]:
                tsmom_rows.append({"date": d, "symbol": s, "signal": 1})
        tsmom_panel = pd.DataFrame(tsmom_rows)
        ret = run_backtest(
            panel, strategy="tsmom_overlay", lookback=60, freq="M", tsmom_panel=tsmom_panel
        )
        assert "daily_return" in ret.columns

    def test_mvo_ml_rejects_missing_panel(self):
        with pytest.raises(ValueError, match="mu_panel"):
            run_backtest(_returns_panel(100, 2), strategy="mvo_ml", lookback=20)

    def test_tsmom_overlay_rejects_missing_panel(self):
        with pytest.raises(ValueError, match="tsmom_panel"):
            run_backtest(_returns_panel(100, 2), strategy="tsmom_overlay", lookback=20)

    def test_rebalance_dates_edge_cases(self):
        idx = pd.bdate_range("2015-01-01", periods=10)
        assert rebalance_dates(idx, "M", warmup=100) == []
        with pytest.raises(ValueError, match="unknown rebalance freq"):
            rebalance_dates(idx, "W", warmup=5)


# ══════════════════════════════════════════════════════════════════
# forecast_panel.py
# ══════════════════════════════════════════════════════════════════


class TestForecastBacktest:
    def test_init(self):
        db = mock.MagicMock()
        bt = ForecastBacktest(universe=["SPY", "QQQ"], macro_db=db, run_date="2025-01-01")
        assert bt.universe == ["SPY", "QQQ"]
        assert bt.run_date == pd.Timestamp("2025-01-01")

    def test_init_no_run_date(self):
        db = mock.MagicMock()
        bt = ForecastBacktest(universe=["SPY"], macro_db=db)
        assert bt.run_date is None

    def test_rolling_window_split(self):
        df = _ohlc_df(100)
        splits = list(ForecastBacktest.rolling_window_split(df, train_size=50, test_size=10))
        assert len(splits) >= 1
        for train_idx, _ in splits:
            assert len(train_idx) == 50

    def test_run_forecast_single_horizon(self):
        df = _ohlc_df(200)
        db = mock.MagicMock()
        db.prices_df.return_value = df
        bt = ForecastBacktest(universe=["SPY"], macro_db=db)
        result = bt.run_forecast(symbol="SPY", train_size=50, test_size=15, horizons=(1,))
        assert "prediction_count" in result
        assert "horizon_results" in result

    def test_run_forecast_missing_data(self):
        db = mock.MagicMock()
        db.prices_df.return_value = None
        bt = ForecastBacktest(universe=["NOPE"], macro_db=db)
        result = bt.run_forecast(symbol="NOPE")
        assert "error" in result

    def test_run_universe_empty(self):
        db = mock.MagicMock()
        db.prices_df_cache = {}
        bt = ForecastBacktest(universe=[], macro_db=db)
        result = bt.run_universe(progress=False)
        assert isinstance(result, dict)

    def test_empty_panel_metrics(self):
        empty = _empty_panel_result()
        assert empty["prediction_count"] == 0
        assert np.isnan(empty["ic"])
        combined = pd.DataFrame({"ensemble_pred": []})
        result = _compute_panel_metrics(combined, pd.Series(dtype=float), {}, "mean")
        assert result["prediction_count"] == 0
