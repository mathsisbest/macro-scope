"""Tests for mmi.ml.features — feature engineering.

Covers feature_columns(), make_features(), _garman_klass_vol(),
and edge cases across default / vol / vol_macro / vol_medium / vol_rich /
vol_rich_plus / mom_rev feature sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmi.ml.features import (
    _garman_klass_vol,
    feature_columns,
    har_feature_names,
    make_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_COLS = {"date", "open", "high", "low", "close", "daily_return"}


def _base_df(rows: int = 50) -> pd.DataFrame:
    """Deterministic OHLCV-like DataFrame with a monotonic close."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = 100.0 * (1 + np.cumsum(np.random.normal(0, 0.01, rows)))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + np.random.normal(0, 0.001, rows)),
            "high": close * (1 + np.abs(np.random.normal(0, 0.002, rows))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.002, rows))),
            "close": close,
            "daily_return": np.random.normal(0.0005, 0.01, rows),
        }
    )


# ---------------------------------------------------------------------------
# feature_columns()
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    def test_default_feature_set(self):
        cols = feature_columns("default")
        assert "ret_lag1" in cols
        assert "roll_mean5" in cols
        assert "roll_std20" in cols
        assert len(cols) == 10  # 4 lags + 6 rolling

    def test_vol_feature_set(self):
        cols = feature_columns("vol")
        assert "gk_vol" in cols
        assert "har_vol_1d" in cols
        assert "rv_trail_63d" in cols
        assert len(cols) > 10

    def test_vol_macro_feature_set(self):
        cols = feature_columns("vol_macro")
        assert "yc_10y2y_lag1" in cols
        assert "cape" in cols
        assert "gld_vol_20d_lag1" in cols

    def test_vol_medium_feature_set(self):
        cols = feature_columns("vol_medium")
        assert "vol_of_vol_5d" in cols
        assert "ret_momentum_63d" in cols
        assert "vix_x_yc_slope" in cols

    def test_vol_rich_feature_set(self):
        cols = feature_columns("vol_rich")
        assert "ret_kurt_20d" in cols
        assert "corr_spy_tlt_20d" in cols
        assert "day_of_week" in cols

    def test_vol_rich_plus_feature_set(self):
        cols = feature_columns("vol_rich_plus")
        assert "cpi_inflation_1y" in cols
        assert "breakeven_inflation_20d" in cols
        assert "spy_gld_spread_20d" in cols

    def test_mom_rev_feature_set(self):
        cols = feature_columns("mom_rev")
        assert "mom_21d" in cols
        assert "mom_accel" in cols
        assert "trend_60d" in cols

    def test_unknown_feature_set_returns_base(self):
        cols = feature_columns("nonexistent")
        assert len(cols) == 10
        assert "ret_lag1" in cols

    def test_no_duplicates_in_any_set(self):
        for fs in [
            "default",
            "vol",
            "vol_macro",
            "vol_medium",
            "vol_rich",
            "vol_rich_plus",
            "mom_rev",
        ]:
            cols = feature_columns(fs)
            assert len(cols) == len(set(cols)), f"duplicates in {fs}: {cols}"


# ---------------------------------------------------------------------------
# har_feature_names()
# ---------------------------------------------------------------------------


class TestHarFeatureNames:
    def test_returns_ordered_list(self):
        names = har_feature_names()
        assert names == ["har_vol_1d", "har_vol_5d", "har_vol_22d"]

    def test_reimportable(self):
        assert har_feature_names() == har_feature_names()


# ---------------------------------------------------------------------------
# _garman_klass_vol()
# ---------------------------------------------------------------------------


class TestGarmanKlassVol:
    def test_known_inputs(self):
        o = pd.Series([100.0, 101.0])
        h = pd.Series([102.0, 103.0])
        l_ = pd.Series([99.0, 100.0])
        c = pd.Series([101.0, 102.0])
        result = _garman_klass_vol(o, h, l_, c)
        assert len(result) == 2
        assert result.notna().all()
        assert (result >= 0).all()

    def test_high_equals_low_returns_zero(self):
        o = pd.Series([100.0])
        h = pd.Series([100.0])
        l_ = pd.Series([100.0])
        c = pd.Series([100.0])
        result = _garman_klass_vol(o, h, l_, c)
        assert result.iloc[0] == 0.0

    def test_open_equals_close_works(self):
        o = pd.Series([100.0, 101.0])
        h = pd.Series([105.0, 106.0])
        l_ = pd.Series([95.0, 96.0])
        c = pd.Series([100.0, 101.0])  # same as open
        result = _garman_klass_vol(o, h, l_, c)
        assert result.notna().all()
        assert result.iloc[0] > 0  # high-low gap produces vol even when O=C

    def test_nan_input_produces_nan(self):
        o = pd.Series([np.nan])
        h = pd.Series([105.0])
        l_ = pd.Series([95.0])
        c = pd.Series([100.0])
        result = _garman_klass_vol(o, h, l_, c)
        assert np.isnan(result.iloc[0])

    def test_all_nan_returns_nan(self):
        o = pd.Series([np.nan])
        h = pd.Series([np.nan])
        l_ = pd.Series([np.nan])
        c = pd.Series([np.nan])
        result = _garman_klass_vol(o, h, l_, c)
        assert np.isnan(result.iloc[0])

    def test_gk_formula_symmetry(self):
        """Symmetric high/low and symmetric close/open yields same result."""
        o = pd.Series([100.0, 100.0])
        h = pd.Series([105.0, 105.0])
        l_ = pd.Series([95.0, 95.0])
        c = pd.Series([100.0, 100.0])  # close == open
        result = _garman_klass_vol(o, h, l_, c)
        assert result.iloc[0] == pytest.approx(result.iloc[1], rel=1e-10)


# ---------------------------------------------------------------------------
# make_features()
# ---------------------------------------------------------------------------


class TestMakeFeatures:
    def test_default_adds_base_features(self):
        df = _base_df(30)
        out = make_features(df, feature_set="default")
        for c in ["ret", "ret_lag1", "roll_mean5", "target_next_ret"]:
            assert c in out.columns, f"missing {c}"
        assert "gk_vol" not in out.columns

    def test_default_has_no_nan_except_lookahead_rows(self):
        df = _base_df(50)
        out = make_features(df, feature_set="default")
        # roll_std20 needs 20 rows; skip generously
        tail = out.iloc[25:]
        for c in ["ret_lag1", "ret_lag5", "roll_mean5", "roll_std20"]:
            assert tail[c].notna().all(), f"NaNs remain in {c}"

    def test_vol_adds_gk_and_har(self):
        df = _base_df(50)
        out = make_features(df, feature_set="vol")
        assert "gk_vol" in out.columns
        assert "har_vol_5d" in out.columns

    def test_vol_without_ohlc_raises_keyerror(self):
        df = _base_df(50).drop(columns=["open", "high", "low"])
        with pytest.raises(KeyError):
            make_features(df, feature_set="vol")

    def test_mom_rev_adds_momentum_features(self):
        df = _base_df(100)
        out = make_features(df, feature_set="mom_rev")
        assert "mom_21d" in out.columns
        assert "rev_5d" in out.columns
        # Sufficient rows means no NaN for window features
        tail = out.iloc[65:]
        assert tail["mom_21d"].notna().all()

    def test_mom_rev_with_short_df(self):
        df = _base_df(10)
        out = make_features(df, feature_set="mom_rev")
        assert "mom_21d" in out.columns  # column exists
        assert out["mom_21d"].isna().all()  # but all NaN (need 21 rows)

    def test_vol_macro_with_macro_df(self):
        df = _base_df(60)
        macro = pd.DataFrame(
            {
                "date": df["date"],
                "T10Y2Y": np.random.uniform(0.0, 1.0, 60),
                "VIXCLS": np.random.uniform(10, 30, 60),
            }
        )
        out = make_features(df, feature_set="vol_macro", macro_df=macro)
        assert "yc_10y2y_lag1" in out.columns
        # Macro merges backward so first rows may be NaN
        assert out["yc_10y2y_lag1"].notna().any()

    def test_vol_macro_without_macro_df(self):
        df = _base_df(60)
        out = make_features(df, feature_set="vol_macro")
        assert "yc_10y2y_lag1" in out.columns
        # Without macro data all macro-specific features are NaN
        assert out["yc_10y2y_lag1"].isna().all()

    def test_vol_macro_with_asset_dfs(self):
        df = _base_df(60).copy()
        df["date"] = pd.date_range("2024-01-01", periods=60, freq="D")
        asset_dfs = {
            "GLD": df[["date", "daily_return"]].copy(),
            "TLT": df[["date", "daily_return"]].copy(),
        }
        out = make_features(
            df,
            feature_set="vol_macro",
            macro_df=pd.DataFrame({"date": df["date"], "T10Y2Y": [0.5] * 60}),
            asset_dfs=asset_dfs,
        )
        assert "gld_vol_20d_lag1" in out.columns

    def test_vol_medium_adds_medium_features(self):
        df = _base_df(100)
        out = make_features(df, feature_set="vol_medium")
        assert "vol_of_vol_5d" in out.columns
        assert "ret_momentum_63d" in out.columns

    def test_vol_rich_adds_rich_features(self):
        df = _base_df(100)
        out = make_features(df, feature_set="vol_rich")
        assert "ret_kurt_20d" in out.columns
        assert "day_of_week" in out.columns

    def test_vol_rich_plus_adds_extended_features(self):
        df = _base_df(100)
        out = make_features(df, feature_set="vol_rich_plus")
        assert "indpro_growth_1y" in out.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "daily_return"])
        out = make_features(df, feature_set="default")
        assert out.empty or "ret" in out.columns

    def test_single_row(self):
        df = _base_df(1)
        out = make_features(df, feature_set="default")
        assert not out.empty
        assert out["ret_lag1"].isna().iloc[0]  # can't lag a single row

    def test_target_next_ret_is_shifted(self):
        df = _base_df(10)
        out = make_features(df, feature_set="default")
        # target_next_ret for row t should be ret at row t+1
        assert out["target_next_ret"].iloc[0] == out["ret"].iloc[1]

    def test_feature_columns_match_make_features_output(self):
        df = _base_df(50)
        for fs in ["default", "vol", "mom_rev"]:
            cols = feature_columns(fs)
            out = make_features(df, feature_set=fs)
            for c in cols:
                assert c in out.columns, f"{fs}: {c} missing from output"
