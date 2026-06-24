"""build_returns_panel pivots correctly; compute covers all strategies; the CLI lands the table."""

import argparse
import logging

import duckdb
import numpy as np
import pandas as pd

import mmi.cli as cli
from mmi.portfolio import windows
from mmi.portfolio.compute import (
    btc_aligned_returns,
    build_returns_panel,
    compute_attribution,
    compute_portfolio_returns,
)


# Use real benchmark tickers (SPY + a bond) so the 60/40 benchmark is produced too.
def _long(n_days: int = 120, assets: tuple = ("SPY", "TLT", "QQQ"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    for asset in assets:
        rets = rng.normal(0.0004, 0.01, n_days)
        for day, ret in zip(idx, rets, strict=True):
            # asset_class mirrors the real fct_asset_daily schema (cmd_portfolio filters on it);
            # all fixture assets are non-crypto so the backtest universe is unchanged.
            rows.append(
                {"symbol": asset, "date": day, "daily_return": ret, "asset_class": "equities"}
            )
    return pd.DataFrame(rows)


def test_build_returns_panel_pivots_to_wide():
    panel = build_returns_panel(_long(10, ("A", "B")))
    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == 10
    assert panel.index.is_monotonic_increasing


def test_compute_portfolio_returns_covers_all_strategies_and_benchmark():
    # include_ml=False keeps this fast (no RF forecast); mvo_ml has its own dedicated test.
    out = compute_portfolio_returns(_long(150), lookback=30, freq="M", cost=0.001, include_ml=False)
    assert set(out["strategy"].unique()) == {
        "equal_weight",
        "inverse_vol",
        "risk_parity",
        "mvo_histmean",
        "sixty_forty",
    }
    assert list(out.columns) == [
        "window_id",
        "strategy",
        "date",
        "daily_return",
        "cumulative_return",
    ]


def test_benchmark_skipped_when_its_tickers_absent():
    out = compute_portfolio_returns(
        _long(150, ("A", "B", "C")), lookback=30, freq="M", include_ml=False
    )
    assert "sixty_forty" not in set(out["strategy"].unique())
    assert {"equal_weight", "inverse_vol", "risk_parity", "mvo_histmean"} == set(
        out["strategy"].unique()
    )


def test_compute_stamps_the_window_dimension():
    # Phase D: every landed frame carries a `window` column; it defaults to the single window D3
    # ships, and an explicit window is honoured (so D6 can run several).
    returns = compute_portfolio_returns(_long(150), lookback=30, freq="M", include_ml=False)
    assert (returns["window_id"] == windows.DEFAULT_WINDOW).all()
    attribution = compute_attribution(_long(150), lookback=30, freq="M", include_ml=False)
    assert "window_id" in attribution.columns
    assert (attribution["window_id"] == windows.DEFAULT_WINDOW).all()
    tagged = compute_portfolio_returns(
        _long(150), lookback=30, freq="M", include_ml=False, window=windows.INC_BTC_2015
    )
    assert (tagged["window_id"] == windows.INC_BTC_2015).all()


def _btc_asset_daily(dates, returns, equity_dates=None):
    """Build a minimal asset_daily DataFrame with BTC rows and optional equity rows."""
    rows = []
    for d, r in zip(dates, returns, strict=True):
        rows.append(
            {"symbol": "BTC", "date": pd.Timestamp(d), "daily_return": r, "asset_class": "crypto"}
        )
    # Add at least one equity asset so equity_dates is non-empty (required by btc_aligned_returns).
    eq_dates = list(equity_dates) if equity_dates is not None else list(dates)
    for d in eq_dates:
        rows.append(
            {
                "symbol": "SPY",
                "date": pd.Timestamp(d),
                "daily_return": 0.001,
                "asset_class": "equities",
            }
        )
    return pd.DataFrame(rows)


def test_btc_aligned_returns_warns_on_interior_nan(caplog):
    """Interior NaN triggers exactly one WARNING with the fill count."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    returns = [0.01] * 4 + [float("nan")] + [0.01] * 5  # 1 interior NaN at position 4
    df = _btc_asset_daily(dates, returns)

    with caplog.at_level(logging.WARNING, logger="portfolio.compute"):
        result = btc_aligned_returns(df)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"Expected 1 warning, got: {[r.message for r in warnings]}"
    assert "1" in warnings[0].message
    # Numeric result is unchanged from fillna(0.0) — verify a non-empty frame is returned.
    assert not result.empty
    assert "daily_return" in result.columns


def test_btc_aligned_returns_no_warning_on_clean_series(caplog):
    """A clean BTC series with no NaN must produce no WARNING log entries."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    returns = [0.01] * 10
    df = _btc_asset_daily(dates, returns)

    with caplog.at_level(logging.WARNING, logger="portfolio.compute"):
        result = btc_aligned_returns(df)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [], f"Unexpected warnings: {[r.message for r in warnings]}"
    assert not result.empty


def test_btc_aligned_returns_warns_with_correct_count(caplog):
    """Multiple interior NaN values: warning message must contain the exact fill count."""
    dates = pd.bdate_range("2020-01-01", periods=15)
    # Positions 3 and 7 are interior NaN (not leading, not trailing)
    returns = [
        0.01,
        0.01,
        0.01,
        float("nan"),
        0.01,
        0.01,
        0.01,
        float("nan"),
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
    ]
    df = _btc_asset_daily(dates, returns)

    with caplog.at_level(logging.WARNING, logger="portfolio.compute"):
        btc_aligned_returns(df)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "2" in warnings[0].message


def test_cmd_portfolio_lands_raw_portfolio_returns(monkeypatch, tmp_path):
    db = tmp_path / "p.duckdb"
    setup = duckdb.connect(str(db))
    setup.execute("create schema if not exists marts")
    setup.register("_fad", _long(320))
    setup.execute("create table marts.fct_asset_daily as select * from _fad")
    setup.close()

    monkeypatch.setattr(cli, "connect", lambda *a, **k: duckdb.connect(str(db)))
    assert cli.cmd_portfolio(argparse.Namespace()) == 0

    check = duckdb.connect(str(db))
    try:
        strategies = check.execute("select distinct strategy from raw.portfolio_returns").fetchall()
    finally:
        check.close()
    assert {s[0] for s in strategies} == {
        "equal_weight",
        "inverse_vol",
        "risk_parity",
        "mvo_histmean",
        "sixty_forty",
        "mvo_ml",
    }
