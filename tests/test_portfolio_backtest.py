"""Backtest: strictly look-ahead-free (window-boundary invariant), cost/drift correct, robust."""

import numpy as np
import pandas as pd
import pytest

import mmi.portfolio.backtest as bt
from mmi.portfolio.backtest import rebalance_dates, run_backtest


def _panel(n_days: int, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    data = rng.normal(0.0004, 0.01, size=(n_days, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_no_lookahead_window_ends_strictly_before_each_rebalance(monkeypatch):
    """The real guard: every covariance window must end strictly before its rebalance date.

    Catches same-bar leaks (e.g. iloc[:-1] -> iloc[:]) and off-by-one (tail(lookback+1)) that the
    tail-truncation check below is structurally blind to.
    """
    panel = _panel(400)
    lookback, freq = 60, "M"
    rebals = sorted(rebalance_dates(panel.index, freq, lookback))
    seen: list[tuple] = []
    original = bt._solve

    def spy(strategy, window):
        seen.append((window.index.max(), len(window)))
        return original(strategy, window)

    monkeypatch.setattr(bt, "_solve", spy)
    run_backtest(panel, strategy="risk_parity", lookback=lookback, freq=freq)

    assert len(seen) == len(rebals)
    for (window_max, window_len), rebal_date in zip(seen, rebals, strict=True):
        assert window_max < rebal_date  # no same-bar leak
        assert window_len == lookback


def test_no_lookahead_truncating_future_keeps_past_unchanged():
    panel = _panel(400)
    full = run_backtest(panel, strategy="risk_parity", lookback=60, freq="M")
    trunc = run_backtest(panel.iloc[:380], strategy="risk_parity", lookback=60, freq="M")
    cutoff = panel.index[350]
    assert np.allclose(
        full.loc[:cutoff, "daily_return"].to_numpy(),
        trunc.loc[:cutoff, "daily_return"].to_numpy(),
    )


def test_first_rebalance_cost_is_half_turnover():
    # entry from cash -> turnover = sum|target - 0| = 1.0 -> drag = cost/2 (round-trip convention)
    panel = _panel(400)
    lb, freq, cost = 60, "M", 0.01
    t0 = sorted(rebalance_dates(panel.index, freq, lb))[0]
    free = run_backtest(panel, strategy="equal_weight", lookback=lb, freq=freq, cost=0.0)
    costly = run_backtest(panel, strategy="equal_weight", lookback=lb, freq=freq, cost=cost)
    drag = float(free.loc[t0, "daily_return"] - costly.loc[t0, "daily_return"])
    assert np.isclose(drag, cost * 0.5)  # catches a dropped 0.5


def test_constant_returns_pin_drift_and_renormalisation():
    idx = pd.bdate_range("2015-01-01", periods=200)
    panel = pd.DataFrame(0.001, index=idx, columns=["A", "B", "C"])  # every asset +0.1%/day
    out = run_backtest(panel, strategy="equal_weight", lookback=60, freq="M", cost=0.0)
    t0 = sorted(rebalance_dates(idx, "M", 60))[0]
    assert np.allclose(out.loc[t0:, "daily_return"].to_numpy(), 0.001)  # catches skip-renormalise


def test_returns_are_clipped_at_minus_one():
    # an impossible <-100% tick must be treated identically to exactly -100% (no phantom leverage)
    p_extreme, p_clipped = _panel(120), _panel(120)
    p_extreme.iloc[80, 0] = -1.5
    p_clipped.iloc[80, 0] = -1.0
    a = run_backtest(p_extreme, strategy="equal_weight", lookback=30, freq="M", cost=0.0)
    b = run_backtest(p_clipped, strategy="equal_weight", lookback=30, freq="M", cost=0.0)
    assert np.allclose(a["daily_return"].to_numpy(), b["daily_return"].to_numpy())


def test_single_asset_panel_runs_all_strategies():
    panel = _panel(120, n_assets=1)
    for strategy in ("equal_weight", "inverse_vol", "risk_parity"):
        out = run_backtest(panel, strategy=strategy, lookback=30, freq="M")
        assert out["daily_return"].notna().all()


def test_zero_variance_asset_does_not_produce_nan():
    panel = _panel(200)
    panel["A0"] = 0.0  # a constant (zero-variance) asset must not NaN-corrupt the run
    out = run_backtest(panel, strategy="inverse_vol", lookback=60, freq="M", cost=0.0)
    assert out["daily_return"].notna().all()
    assert np.isfinite(out["cumulative_return"].to_numpy()).all()


def test_rebalance_dates_are_last_trading_day_of_each_month():
    idx = pd.bdate_range("2015-01-01", periods=400)
    dates = rebalance_dates(idx, "M", warmup=60)
    eligible = idx[60:]
    for d in dates:
        in_month = eligible[(eligible.year == d.year) & (eligible.month == d.month)]
        assert d == in_month.max()  # last available trading day of its month
    assert len(dates) == len({(d.year, d.month) for d in dates})


def test_rebalance_quarterly_is_sparser_than_monthly():
    idx = pd.bdate_range("2015-01-01", periods=400)
    assert len(rebalance_dates(idx, "Q", 60)) < len(rebalance_dates(idx, "M", 60))


def test_output_shape_and_columns():
    out = run_backtest(_panel(300), strategy="equal_weight", lookback=60, freq="M")
    assert list(out.columns) == ["daily_return", "cumulative_return"]
    assert out.index.is_monotonic_increasing


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        run_backtest(_panel(100), strategy="nope", lookback=20)
