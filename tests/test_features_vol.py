"""Tests for the optional volatility feature set in mmi.ml.features (task C2).

Assertions:
  1. Default feature_columns() is UNCHANGED when feature_set is omitted / 'default'.
  2. Truncation-invariance: truncating future rows leaves earlier feature values identical.
  3. Garman-Klass values are finite on a toy OHLC frame.
  4. Vol feature names only appear when feature_set='vol'.
  5. All vol features at row t use only data from rows <= t (leakage-free).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mmi.ml.features import (
    _vol_feature_names,
    feature_columns,
    make_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_COLS_EXPECTED = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5"] + [
    "roll_mean5",
    "roll_std5",
    "roll_mean10",
    "roll_std10",
    "roll_mean20",
    "roll_std20",
]


def _make_ohlc_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLC + daily_return dataframe."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    daily_return = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
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


# ---------------------------------------------------------------------------
# Test 1: default feature_columns() unchanged
# ---------------------------------------------------------------------------


def test_feature_columns_default_unchanged() -> None:
    """feature_columns() with no args must match the legacy set exactly."""
    assert feature_columns() == _DEFAULT_COLS_EXPECTED


def test_feature_columns_explicit_default_unchanged() -> None:
    """feature_columns(feature_set='default') must also match the legacy set."""
    assert feature_columns(feature_set="default") == _DEFAULT_COLS_EXPECTED


def test_feature_columns_vol_is_superset() -> None:
    """feature_columns(feature_set='vol') starts with the default set, then adds vol names."""
    vol_cols = feature_columns(feature_set="vol")
    default_cols = feature_columns()
    # Default prefix preserved
    assert vol_cols[: len(default_cols)] == default_cols
    # Extra vol names appended
    extra = vol_cols[len(default_cols) :]
    assert extra == _vol_feature_names()
    assert len(extra) > 0


# ---------------------------------------------------------------------------
# Test 2: truncation-invariance
# ---------------------------------------------------------------------------


def test_truncation_invariance_default() -> None:
    """Truncating future rows must not change earlier feature values (default set)."""
    df = _make_ohlc_df(n=60)
    full = make_features(df.copy())
    trunc = make_features(df.iloc[:40].copy())

    cols = feature_columns()
    for col in cols:
        pd.testing.assert_series_equal(
            full[col].iloc[:40].reset_index(drop=True),
            trunc[col].reset_index(drop=True),
            check_names=False,
            obj=f"col={col}",
        )


def test_truncation_invariance_vol() -> None:
    """Truncating future rows must not change earlier vol feature values."""
    df = _make_ohlc_df(n=60)
    full = make_features(df.copy(), feature_set="vol")
    trunc = make_features(df.iloc[:40].copy(), feature_set="vol")

    cols = feature_columns(feature_set="vol")
    for col in cols:
        pd.testing.assert_series_equal(
            full[col].iloc[:40].reset_index(drop=True),
            trunc[col].reset_index(drop=True),
            check_names=False,
            obj=f"col={col}",
        )


# ---------------------------------------------------------------------------
# Test 3: Garman-Klass values finite on toy OHLC frame
# ---------------------------------------------------------------------------


def test_garman_klass_finite() -> None:
    """gk_vol column must be non-NaN and non-infinite for well-formed OHLC data."""
    df = _make_ohlc_df(n=30)
    out = make_features(df.copy(), feature_set="vol")
    gk = out["gk_vol"]
    assert gk.notna().all(), "gk_vol must not contain NaN on valid OHLC data"
    assert np.isfinite(gk).all(), "gk_vol must be finite"
    assert (gk >= 0).all(), "gk_vol must be non-negative"


def test_garman_klass_manual_scalar() -> None:
    """Spot-check GK formula on a known scalar."""
    # H=110, L=90, C=105, O=100
    # log(H/L) = log(110/90), log(C/O) = log(105/100)
    log_hl = math.log(110 / 90)
    log_co = math.log(105 / 100)
    c1 = 0.5
    c2 = 2.0 * math.log(2) - 1.0
    expected = math.sqrt(c1 * log_hl**2 - c2 * log_co**2)

    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=1, freq="B"),
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "daily_return": [0.05],
        }
    )
    out = make_features(df.copy(), feature_set="vol")
    assert pytest.approx(out["gk_vol"].iloc[0], rel=1e-9) == expected


# ---------------------------------------------------------------------------
# Test 4: vol features absent when feature_set='default'
# ---------------------------------------------------------------------------


def test_vol_features_absent_in_default() -> None:
    """make_features() with default set must not add any vol columns."""
    df = _make_ohlc_df(n=30)
    out = make_features(df.copy())
    vol_names = set(_vol_feature_names())
    overlap = vol_names & set(out.columns)
    assert not overlap, f"Unexpected vol columns in default output: {overlap}"


# ---------------------------------------------------------------------------
# Test 5: leakage-free — features at row t use only data from rows <= t
# ---------------------------------------------------------------------------


def test_vol_features_leakage_free() -> None:
    """Replacing future rows with NaN must not alter any vol feature at earlier rows."""
    df = _make_ohlc_df(n=50)
    cut = 30  # check rows 0..29

    # Poison future rows: replace OHLC with NaN beyond cut
    df_poisoned = df.copy()
    df_poisoned.loc[cut:, ["open", "high", "low", "close", "daily_return"]] = float("nan")

    clean = make_features(df.iloc[:cut].copy(), feature_set="vol")
    poisoned = make_features(df_poisoned.copy(), feature_set="vol")

    vol_cols = _vol_feature_names()
    for col in vol_cols:
        pd.testing.assert_series_equal(
            clean[col].reset_index(drop=True),
            poisoned[col].iloc[:cut].reset_index(drop=True),
            check_names=False,
            obj=f"leakage check col={col}",
        )
