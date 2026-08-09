"""Attribution: per-asset contributions reconcile to the gross return; risk shares sum to 1."""

import numpy as np
import pandas as pd

from mmi.portfolio.backtest import run_backtest, run_backtest_full
from mmi.portfolio.compute import compute_attribution


def _long(n: int = 300, assets: tuple = ("SPY", "TLT", "QQQ"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    rows = []
    for asset in assets:
        rets = rng.normal(0.0004, 0.01, n)
        for day, ret in zip(idx, rets, strict=True):
            rows.append({"symbol": asset, "date": day, "daily_return": float(ret)})
    return pd.DataFrame(rows)


def _panel(n: int = 300, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    data = rng.normal(0.0004, 0.01, size=(n, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_run_backtest_is_the_returns_half_of_run_backtest_full():
    panel = _panel(200)
    only = run_backtest(panel, strategy="risk_parity", lookback=60, freq="M")
    both = run_backtest_full(panel, strategy="risk_parity", lookback=60, freq="M")[0]
    pd.testing.assert_frame_equal(only, both)


def test_contributions_reconcile_to_net_daily_return():
    panel = _panel(200)
    returns, contrib = run_backtest_full(
        panel, strategy="equal_weight", lookback=60, freq="M", cost=0.01
    )
    assets = [c for c in contrib.columns if c != "__cost__"]
    # gross (sum of per-asset contributions) + the cost column == the recorded net daily return
    net = contrib[assets].sum(axis=1) + contrib["__cost__"]
    aligned = returns.loc[contrib.index, "daily_return"]
    assert np.allclose(net.to_numpy(), aligned.to_numpy())


def test_attribution_return_reconciles_and_risk_sums_to_one():
    attr = compute_attribution(_long(300))
    assert not attr.empty
    for _, grp in attr.groupby("strategy"):
        assert np.isclose(grp["contribution_to_risk"].sum(), 1.0, atol=1e-9)


def test_attribution_includes_all_input_assets():
    attr = compute_attribution(_long(300, ("SPY", "TLT", "QQQ")))
    assert not attr.empty
    assert set(attr["symbol"]) == {"SPY", "TLT", "QQQ"}


def test_attribution_contributions_are_finite():
    attr = compute_attribution(_long(320))
    assert not attr.empty
    assert attr["contribution_to_return"].notna().all()


def test_per_asset_costs_exact_calculation():
    panel = _panel(400)
    asset_costs = {"A0": 0.005, "A1": 0.01, "A2": 0.02}
    # Initial rebalance to equal weight (1/3 per asset):
    # turnover = 1.0 (each asset goes 0 -> 1/3, total 1.0)
    # total cost = 0.5 * 1/3 * ( (0.005 + 0.002) + (0.01 + 0.002) + (0.02 + 0.002) )
    #            = 0.5 * 1/3 * ( 0.007 + 0.012 + 0.022 ) = 0.5 * 1/3 * 0.041 = 0.0068333...
    _, contrib = run_backtest_full(
        panel,
        strategy="equal_weight",
        lookback=60,
        freq="M",
        cost=0.001,
        asset_costs=asset_costs,
        slippage=0.002,
    )
    first_rebal_cost = contrib["__cost__"].iloc[0]
    np.testing.assert_allclose(first_rebal_cost, -0.006833333333333334, atol=1e-6)


def test_per_asset_costs_and_slippage_differential():
    panel = _panel(400)
    # Default cost (cost=0.001, no asset_costs, no slippage)
    _, contrib_default = run_backtest_full(
        panel, strategy="equal_weight", lookback=60, freq="M", cost=0.001
    )
    # High custom asset costs + slippage
    asset_costs = {"A0": 0.005, "A1": 0.01, "A2": 0.02}
    _, contrib_custom = run_backtest_full(
        panel,
        strategy="equal_weight",
        lookback=60,
        freq="M",
        cost=0.001,
        asset_costs=asset_costs,
        slippage=0.002,
    )
    # Total cost paid under custom high-cost config must be significantly higher
    total_cost_default = contrib_default["__cost__"].sum()
    total_cost_custom = contrib_custom["__cost__"].sum()
    assert total_cost_custom < total_cost_default  # negative values, so custom is more negative
    np.testing.assert_allclose(total_cost_default, -0.000707, atol=1e-4)
    np.testing.assert_allclose(total_cost_custom, -0.009737, atol=1e-4)
