"""ML and regime edge-case tests (Wave 3a, task C9).

Covers:
  1. forecast.train_and_backtest raises ValueError (not crash) on <60 obs, single-row,
     and all-constant series — verifying the pipeline guard without importing sklearn at module
     scope (the function itself handles it before any training).
  2. regime.label_regimes on empty vol_20d (returns empty frame, no raise).
  3. regime.label_regimes on constant vol_20d (qcut tie handling via rank, no raise).
  4. forecast_panel.walk_forward_mu short-panel min_train path returns empty/zero-skill
     DataFrames without crashing.

No overlap with test_features.py / test_forecast_leakage.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmi.ml.forecast import train_and_backtest
from mmi.ml.forecast_panel import walk_forward_mu
from mmi.ml.regime import label_regimes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCon:
    """Minimal DB connection stub that returns a fixed DataFrame via .execute().df()."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def execute(self, _sql: str, _params=None):
        return self

    def df(self) -> pd.DataFrame:
        return self._df.copy()


def _asset_df(n: int, *, symbol: str = "SPY", constant_return: float | None = None) -> pd.DataFrame:
    """Synthetic asset DataFrame with columns [date, close, daily_return, symbol].

    When ``constant_return`` is given every row has the same return value (all-constant series).
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n)
    if constant_return is not None:
        rets = np.full(n, constant_return)
    else:
        rets = rng.normal(0.0004, 0.01, n)
    close = 100.0 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "daily_return": rets,
            "symbol": symbol,
        }
    )


# ---------------------------------------------------------------------------
# 1. train_and_backtest ValueError guard
# ---------------------------------------------------------------------------


def test_train_and_backtest_raises_on_too_few_obs():
    """Less than 60 usable observations must raise ValueError, not crash the process."""
    df = _asset_df(30)  # 30 rows; after dropna on features will be well below 60
    con = _MockCon(df)
    with pytest.raises(ValueError, match="not enough observations"):
        train_and_backtest(con, symbol="SPY")


def test_train_and_backtest_raises_on_single_row():
    """A single-row frame is the extreme of the too-few-obs case; ValueError expected."""
    df = _asset_df(1)
    con = _MockCon(df)
    with pytest.raises(ValueError, match="not enough observations"):
        train_and_backtest(con, symbol="SPY")


def test_train_and_backtest_raises_on_all_constant_series():
    """All-constant return series → features are all-constant → still too few obs after dropna.

    The guard is an observation-count check; a constant series collapses features via NaN
    or produces degenerate feature rows, so len(y) < 60 must trigger cleanly.
    """
    df = _asset_df(50, constant_return=0.001)  # 50 rows, well-formed but constant
    con = _MockCon(df)
    with pytest.raises(ValueError, match="not enough observations"):
        train_and_backtest(con, symbol="SPY")


# ---------------------------------------------------------------------------
# 2. regime.label_regimes — empty input
# ---------------------------------------------------------------------------


def _regime_con(rows: list[dict]) -> _MockCon:
    """Build a stub returning a vol_20d frame for label_regimes."""
    df = pd.DataFrame(rows, columns=["symbol", "date", "vol_20d"])
    return _MockCon(df)


def test_label_regimes_empty_returns_empty_frame():
    """Empty vol_20d input must return an empty DataFrame with the correct columns, no raise."""
    con = _regime_con([])
    result = label_regimes(con)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert set(result.columns) == {"symbol", "date", "vol_20d", "regime"}


# ---------------------------------------------------------------------------
# 3. regime.label_regimes — constant vol_20d (qcut tie handling)
# ---------------------------------------------------------------------------


def test_label_regimes_constant_vol_no_raise():
    """Constant vol_20d per symbol must not raise (rank(method='first') breaks ties)."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    rows = [{"symbol": "AAA", "date": d, "vol_20d": 0.01} for d in dates]
    con = _regime_con(rows)
    result = label_regimes(con)
    assert not result.empty
    assert set(result.columns) == {"symbol", "date", "vol_20d", "regime"}
    assert result["regime"].notna().all()


def test_label_regimes_constant_vol_multi_symbol_no_raise():
    """Two symbols with identical constant vol each receive valid regime labels."""
    dates = pd.bdate_range("2020-01-01", periods=9)
    rows = [{"symbol": "AAA", "date": d, "vol_20d": 0.02} for d in dates] + [
        {"symbol": "BBB", "date": d, "vol_20d": 0.02} for d in dates
    ]
    con = _regime_con(rows)
    result = label_regimes(con)
    assert not result.empty
    assert result["regime"].notna().all()
    for sym in ("AAA", "BBB"):
        assert sym in result["symbol"].to_numpy()


# ---------------------------------------------------------------------------
# 4. forecast_panel.walk_forward_mu — short panel / min_train path
# ---------------------------------------------------------------------------


def _short_panel(n: int = 30, assets: tuple[str, ...] = ("SPY",)) -> pd.DataFrame:
    """Build a short asset_daily DataFrame (far fewer rows than min_train)."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2022-01-01", periods=n)
    rows = []
    for sym in assets:
        rets = rng.normal(0.0003, 0.009, n)
        for d, r in zip(idx, rets, strict=True):
            rows.append({"symbol": sym, "date": d, "daily_return": float(r)})
    return pd.DataFrame(rows)


def test_walk_forward_mu_short_panel_returns_empty_frames():
    """When every symbol has fewer rows than min_train, both returned DataFrames must be empty."""
    df = _short_panel(n=30)
    rebals = pd.bdate_range("2022-02-01", periods=5, freq="3W")
    mu, skill = walk_forward_mu(df, rebals, horizon=5, min_train=120, n_estimators=20, seed=0)

    assert isinstance(mu, pd.DataFrame)
    assert isinstance(skill, pd.DataFrame)
    assert mu.empty, f"Expected empty mu panel, got {len(mu)} rows"
    assert skill.empty, f"Expected empty skill frame, got {len(skill)} rows"


def test_walk_forward_mu_short_panel_multi_asset_empty_no_raise():
    """Multi-asset short panel skips all assets cleanly — no exception, empty results."""
    df = _short_panel(n=25, assets=("AAA", "BBB", "CCC"))
    rebals = pd.bdate_range("2022-02-01", periods=3, freq="ME")
    mu, skill = walk_forward_mu(df, rebals, horizon=10, min_train=200, n_estimators=10, seed=0)

    assert mu.empty
    assert skill.empty


def test_walk_forward_mu_zero_rebal_dates_returns_empty():
    """Passing an empty rebalance date list returns empty frames without raising."""
    df = _short_panel(n=200)
    mu, skill = walk_forward_mu(df, [], horizon=5, min_train=60, n_estimators=10, seed=0)

    assert mu.empty
    assert skill.empty
