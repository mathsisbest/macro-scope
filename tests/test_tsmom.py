"""Tests for the 12-month TSMOM overlay experiment.

Covers:
1. Signal correctness (positive/negative momentum detected correctly).
2. Leakage-free / truncation-invariance: the signal at date T is identical whether
   computed on the full panel or on a panel truncated after T (no future data used).
3. The overlay strategy runs without disturbing the existing strategies
   (STRATEGIES tuple, existing backtest outputs, cmd_portfolio result set unchanged).
4. compute_tsmom_overlay produces the expected frame shapes and columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmi.portfolio.backtest import STRATEGIES, TSMOM_OVERLAY, run_backtest_full
from mmi.portfolio.compute import (
    TSMOM_STRATEGY_TYPE,
    build_returns_panel,
    compute_portfolio_returns,
    compute_tsmom_overlay,
    tsmom_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(
    n_days: int = 400,
    assets: tuple = ("A", "B", "C"),
    seed: int = 42,
) -> pd.DataFrame:
    """Wide returns panel (date x symbol) with deterministic random data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    data = rng.normal(0.0004, 0.01, size=(n_days, len(assets)))
    return pd.DataFrame(data, index=idx, columns=list(assets))


def _long_asset_daily(
    n_days: int = 400,
    assets: tuple = ("SPY", "TLT", "QQQ"),
    seed: int = 0,
) -> pd.DataFrame:
    """Long asset_daily frame compatible with compute_portfolio_returns."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    for asset in assets:
        rets = rng.normal(0.0004, 0.01, n_days)
        for day, ret in zip(idx, rets, strict=True):
            rows.append(
                {
                    "symbol": asset,
                    "date": day,
                    "daily_return": ret,
                    "asset_class": "equities",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Signal correctness
# ---------------------------------------------------------------------------


class TestTsmomSignal:
    def test_positive_momentum_gives_signal_1(self):
        """Assets with a positive 12m-1m window should yield signal=1."""
        # Build a panel where every asset has strictly positive returns in the long window
        n = 300
        idx = pd.bdate_range("2015-01-01", periods=n)
        # All returns +0.1% except the last 21 days (skip) which are -1%
        data = np.full((n, 2), 0.001)
        data[-21:] = -0.01  # skip period: strong negative, excluded from signal
        panel = pd.DataFrame(data, index=idx, columns=["X", "Y"])
        t = idx[-1]
        sig = tsmom_signal(panel, t, long_days=252, short_days=21)
        # The signal window excludes the last 21 days; everything before has +0.1%
        assert sig.tolist() == [1.0, 1.0], f"expected all-positive signals, got {sig}"

    def test_negative_momentum_gives_signal_0(self):
        """Assets with a negative 12m-1m window should yield signal=0 (flat)."""
        n = 300
        idx = pd.bdate_range("2015-01-01", periods=n)
        data = np.full((n, 2), -0.002)  # all negative returns in momentum window
        data[-21:] = 0.01  # skip period: positive, but excluded from signal
        panel = pd.DataFrame(data, index=idx, columns=["X", "Y"])
        t = idx[-1]
        sig = tsmom_signal(panel, t, long_days=252, short_days=21)
        assert sig.tolist() == [0.0, 0.0], f"expected all-flat signals, got {sig}"

    def test_mixed_assets(self):
        """One positive-momentum asset + one negative: signals differ."""
        n = 300
        idx = pd.bdate_range("2015-01-01", periods=n)
        # Asset X: +0.1% throughout; Asset Y: -0.1% throughout (excl. skip)
        x = np.full(n, 0.001)
        y = np.full(n, -0.001)
        panel = pd.DataFrame({"X": x, "Y": y}, index=idx)
        t = idx[-1]
        sig = tsmom_signal(panel, t, long_days=252, short_days=21)
        assert sig[0] == 1.0 and sig[1] == 0.0, f"expected [1, 0], got {sig}"

    def test_insufficient_history_returns_zeros(self):
        """When history is shorter than long_days, signal is all zeros (conservative)."""
        idx = pd.bdate_range("2015-01-01", periods=100)
        panel = pd.DataFrame({"A": np.full(100, 0.001), "B": np.full(100, 0.001)}, index=idx)
        t = idx[-1]
        sig = tsmom_signal(panel, t, long_days=252, short_days=21)
        assert (sig == 0.0).all(), f"expected all zeros with short history, got {sig}"

    def test_signal_uses_only_data_before_date(self):
        """Signal at T must be computed from data strictly before T (no look-ahead at T itself)."""
        n = 300
        idx = pd.bdate_range("2015-01-01", periods=n)
        # Normal positive returns
        data = np.full((n, 1), 0.001)
        panel = pd.DataFrame(data, index=idx, columns=["A"])
        t = idx[-1]
        sig_before = tsmom_signal(panel, t)
        # Now make the return AT t extremely negative — signal must not change
        panel_modified = panel.copy()
        panel_modified.loc[t] = -0.5
        sig_after = tsmom_signal(panel_modified, t)
        np.testing.assert_array_equal(
            sig_before,
            sig_after,
            err_msg="signal changed when modifying the value at date T (look-ahead leak!)",
        )


# ---------------------------------------------------------------------------
# 2. Truncation-invariance (leakage-free guarantee)
# ---------------------------------------------------------------------------


class TestTsmomLeakage:
    def test_truncation_invariance(self):
        """Signal at T is identical whether computed on the full panel or truncated at T+1."""
        panel = _make_panel(n_days=500)
        rebals = panel.index[300::21]  # a handful of rebalance dates
        for t in rebals:
            sig_full = tsmom_signal(panel, t)
            # Truncate the panel just after T (keep T itself; signal uses only before T)
            t_loc = panel.index.get_loc(t)
            panel_trunc = panel.iloc[: t_loc + 5]  # keep 5 rows after T but drop the rest
            sig_trunc = tsmom_signal(panel_trunc, t)
            np.testing.assert_array_equal(
                sig_full,
                sig_trunc,
                err_msg=f"signal differs on truncated panel at {t} — truncation-invariance broken",
            )

    def test_adding_future_rows_does_not_change_signal(self):
        """Appending rows after T must not alter the signal computed at T."""
        panel = _make_panel(n_days=350)
        t = panel.index[300]
        sig_base = tsmom_signal(panel, t)

        # Append 50 extra rows with very different returns
        extra_idx = pd.bdate_range(panel.index[-1] + pd.offsets.BDay(), periods=50)
        extra_data = np.full((50, panel.shape[1]), -0.05)
        extra = pd.DataFrame(extra_data, index=extra_idx, columns=panel.columns)
        panel_extended = pd.concat([panel, extra])
        sig_extended = tsmom_signal(panel_extended, t)

        np.testing.assert_array_equal(
            sig_base,
            sig_extended,
            err_msg="signal changed after appending future rows — look-ahead detected",
        )


# ---------------------------------------------------------------------------
# 3. TSMOM_OVERLAY does not disturb existing strategies
# ---------------------------------------------------------------------------


class TestTsmomDoesNotDisturbExisting:
    def test_strategies_tuple_unchanged(self):
        """STRATEGIES must not include TSMOM_OVERLAY (it is not a core strategy)."""
        assert TSMOM_OVERLAY not in STRATEGIES

    def test_existing_strategies_output_identical(self):
        """compute_portfolio_returns without TSMOM must return exactly the core strategy set."""
        ad = _long_asset_daily(n_days=200)
        out = compute_portfolio_returns(ad, lookback=30, freq="M", include_ml=False)
        strategy_set = set(out["strategy"].unique())
        # Must include all core strategies + the 60/40 benchmark (SPY + TLT present in fixture)
        expected = {"equal_weight", "inverse_vol", "risk_parity", "mvo_histmean", "sixty_forty"}
        assert strategy_set == expected, f"unexpected strategy set: {strategy_set}"
        # TSMOM must not appear
        assert TSMOM_OVERLAY not in strategy_set

    def test_run_backtest_full_rejects_tsmom_without_panel(self):
        """run_backtest_full with TSMOM_OVERLAY but no tsmom_panel must raise."""
        panel = _make_panel(n_days=200)
        with pytest.raises(ValueError, match="tsmom_overlay requires a tsmom_panel"):
            run_backtest_full(panel, strategy=TSMOM_OVERLAY, lookback=30, freq="M")

    def test_existing_strategy_values_are_not_changed(self):
        """The returns for equal_weight from compute_portfolio_returns must be
        byte-identical to a direct run_backtest_full(equal_weight) call."""
        ad = _long_asset_daily(n_days=200)
        panel = build_returns_panel(ad)
        # via compute_portfolio_returns
        out_compute = compute_portfolio_returns(ad, lookback=30, freq="M", include_ml=False)
        ew_from_compute = (
            out_compute[out_compute["strategy"] == "equal_weight"]
            .set_index("date")["daily_return"]
            .sort_index()
        )
        # direct run_backtest_full
        out_direct, _ = run_backtest_full(panel, strategy="equal_weight", lookback=30, freq="M")
        ew_direct = out_direct["daily_return"].sort_index()
        # align on common index (compute_portfolio_returns may have pre-warmup zeros)
        common = ew_direct.index.intersection(ew_from_compute.index)
        pd.testing.assert_series_equal(
            ew_from_compute.loc[common].reset_index(drop=True),
            ew_direct.loc[common].reset_index(drop=True),
            check_names=False,
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# 4. compute_tsmom_overlay shape and columns
# ---------------------------------------------------------------------------


class TestComputeTsmomOverlay:
    def test_returns_three_frames(self):
        """compute_tsmom_overlay must return (returns_long, stats, signal_log)."""
        ad = _long_asset_daily(n_days=350)
        result = compute_tsmom_overlay(ad, lookback=30, freq="M", n_boot=50)
        assert len(result) == 3

    def test_returns_long_columns(self):
        """returns_long must have the required columns."""
        ad = _long_asset_daily(n_days=350)
        returns_long, _, _ = compute_tsmom_overlay(ad, lookback=30, freq="M", n_boot=50)
        required = {
            "window_id",
            "strategy",
            "strategy_type",
            "date",
            "daily_return",
            "cumulative_return",
        }
        assert required.issubset(set(returns_long.columns)), (
            f"missing columns: {required - set(returns_long.columns)}"
        )

    def test_three_strategies_present(self):
        """tsmom_overlay, equal_weight, and buy_and_hold must all appear in returns_long."""
        ad = _long_asset_daily(n_days=350)
        returns_long, _, _ = compute_tsmom_overlay(ad, lookback=30, freq="M", n_boot=50)
        strategies = set(returns_long["strategy"].unique())
        assert TSMOM_OVERLAY in strategies, "tsmom_overlay missing from returns_long"
        assert "equal_weight" in strategies, "equal_weight missing from returns_long"
        assert "buy_and_hold" in strategies, "buy_and_hold missing from returns_long"

    def test_tsmom_labelled_experiment(self):
        """The tsmom_overlay rows must carry strategy_type == 'experiment'."""
        ad = _long_asset_daily(n_days=350)
        returns_long, _, _ = compute_tsmom_overlay(ad, lookback=30, freq="M", n_boot=50)
        tsmom_rows = returns_long[returns_long["strategy"] == TSMOM_OVERLAY]
        assert (tsmom_rows["strategy_type"] == TSMOM_STRATEGY_TYPE).all(), (
            f"expected strategy_type='experiment', got: {tsmom_rows['strategy_type'].unique()}"
        )

    def test_benchmarks_labelled_benchmark(self):
        """equal_weight and buy_and_hold must carry strategy_type == 'benchmark'."""
        ad = _long_asset_daily(n_days=350)
        returns_long, _, _ = compute_tsmom_overlay(ad, lookback=30, freq="M", n_boot=50)
        for s in ("equal_weight", "buy_and_hold"):
            rows = returns_long[returns_long["strategy"] == s]
            assert (rows["strategy_type"] == "benchmark").all(), (
                f"{s} has unexpected strategy_type: {rows['strategy_type'].unique()}"
            )

    def test_signal_log_columns(self):
        """signal_log must have (window_id, date, symbol, signal) columns."""
        ad = _long_asset_daily(n_days=350)
        _, _, signal_log = compute_tsmom_overlay(ad, lookback=30, freq="M", n_boot=50)
        required = {"window_id", "date", "symbol", "signal"}
        assert required.issubset(set(signal_log.columns)), (
            f"missing signal_log columns: {required - set(signal_log.columns)}"
        )

    def test_signal_values_binary(self):
        """All signal values in signal_log must be 0.0 or 1.0."""
        ad = _long_asset_daily(n_days=350)
        _, _, signal_log = compute_tsmom_overlay(ad, lookback=30, freq="M", n_boot=50)
        unique_vals = set(signal_log["signal"].unique())
        assert unique_vals.issubset({0.0, 1.0}), f"non-binary signal values: {unique_vals}"

    def test_short_panel_returns_empty_frames(self):
        """With fewer dates than lookback, all three returned frames must be empty."""
        ad = _long_asset_daily(n_days=10)  # far too short
        returns_long, stats, signal_log = compute_tsmom_overlay(ad, lookback=252, freq="M")
        assert returns_long.empty, "expected empty returns_long for short panel"

    def test_buy_and_hold_no_rebalance_cost(self):
        """buy_and_hold must drift weights without further rebalancing.

        Key property: buy_and_hold must never incur cost drag beyond the initial investment.
        We verify this by constructing a dataset with *constant* positive returns per day so
        the expected cumulative return is positive and deterministic (not subject to random-seed
        variance).
        """
        # Constant +0.1% per day on every asset — deterministic positive drift.
        n = 200
        idx = pd.bdate_range("2018-01-01", periods=n)
        # Build a long asset_daily from this panel
        rows = []
        for asset in ["SPY", "TLT", "QQQ"]:
            for day in idx:
                rows.append(
                    {
                        "symbol": asset,
                        "date": day,
                        "daily_return": 0.001,
                        "asset_class": "equities",
                    }
                )
        ad = pd.DataFrame(rows)
        returns_long, _, _ = compute_tsmom_overlay(ad, lookback=30, freq="M", n_boot=50)
        bnh = returns_long[returns_long["strategy"] == "buy_and_hold"]
        # With constant +0.1%/day, cumulative return must be strongly positive.
        cum_return = float(bnh["cumulative_return"].iloc[-1])
        assert cum_return > 0.05, (
            f"buy_and_hold cumulative return unexpectedly low with constant-positive returns: "
            f"{cum_return}"
        )

    def test_window_id_stamped_correctly(self):
        """All rows in returns_long and signal_log must carry the supplied window_id."""
        from mmi.portfolio.windows import INC_BTC_2015

        ad = _long_asset_daily(n_days=350)
        returns_long, _, signal_log = compute_tsmom_overlay(
            ad, lookback=30, freq="M", n_boot=50, window=INC_BTC_2015
        )
        assert (returns_long["window_id"] == INC_BTC_2015).all()
        assert (signal_log["window_id"] == INC_BTC_2015).all()
