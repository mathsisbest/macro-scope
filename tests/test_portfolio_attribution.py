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
    # include_ml=False keeps this fast; mvo_ml reconciliation has its own test below.
    attr = compute_attribution(_long(300), lookback=60, freq="M", cost=0.001, include_ml=False)
    assert not attr.empty
    for _, grp in attr.groupby("strategy"):
        assets = grp[grp["symbol"] != "(costs)"]
        gross = grp["strategy_gross_return"].iloc[0]
        assert np.isclose(assets["contribution_to_return"].sum(), gross, atol=1e-9)
        assert np.isclose(assets["contribution_to_risk"].sum(), 1.0, atol=1e-9)
        assert (grp["symbol"] == "(costs)").any()  # the cost row is always present


def test_attribution_omits_never_held_assets_for_the_benchmark():
    # 60/40 holds only SPY + a bond; QQQ is never held -> no attribution row for sixty_forty.
    attr = compute_attribution(
        _long(300, ("SPY", "TLT", "QQQ")), lookback=60, freq="M", include_ml=False
    )
    held = set(attr[attr["strategy"] == "sixty_forty"]["symbol"])
    assert {"SPY", "TLT", "(costs)"} <= held
    assert "QQQ" not in held


def test_mvo_ml_attribution_reconciles():
    attr = compute_attribution(_long(320), lookback=60, freq="M", horizon=5, include_ml=True)
    ml = attr[attr["strategy"] == "mvo_ml"]
    assert not ml.empty
    assets = ml[ml["symbol"] != "(costs)"]
    gross = ml["strategy_gross_return"].iloc[0]
    assert np.isclose(assets["contribution_to_return"].sum(), gross, atol=1e-9)
    assert np.isclose(assets["contribution_to_risk"].sum(), 1.0, atol=1e-9)
