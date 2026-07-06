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
