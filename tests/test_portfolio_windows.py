"""Phase D window construction: BTC calendar-fold, per-window universe/floor, no merged-panel trap.

These guard the riskiest part of the BTC backtest — that ex-BTC 2002+ history is NOT truncated to
BTC's inception, and that BTC's 7-day returns are correctly folded onto the equity calendar.
"""

import argparse

import duckdb
import numpy as np
import pandas as pd

import mmi.cli as cli
from mmi.portfolio import compute, windows


def _equities(symbols, dates) -> list[dict]:
    rng = np.random.default_rng(0)
    return [
        {"symbol": s, "date": d, "daily_return": float(r), "asset_class": "equities"}
        for s in symbols
        for d, r in zip(dates, rng.normal(0.0004, 0.01, len(dates)), strict=True)
    ]


def test_btc_aligned_returns_folds_intervening_days_onto_the_equity_calendar():
    # Equities trade Thu/Fri/Mon; BTC trades every calendar day. BTC +10% each day Fri..Mon.
    eq_dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    btc_dates = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"]
    )
    rows = [
        {"symbol": "SPY", "date": d, "daily_return": 0.01, "asset_class": "equities"}
        for d in eq_dates
    ]
    rows += [
        {"symbol": "BTC", "date": d, "daily_return": r, "asset_class": "crypto"}
        for d, r in zip(btc_dates, [np.nan, 0.10, 0.10, 0.10, 0.10], strict=True)
    ]
    aligned = compute.btc_aligned_returns(pd.DataFrame(rows))

    assert list(aligned["date"]) == list(eq_dates)  # only equity-calendar dates, no weekends
    mon = aligned.loc[aligned["date"] == "2020-01-06", "daily_return"].iloc[0]
    assert np.isclose(mon, 1.10**3 - 1)  # Sat+Sun+Mon fold into Monday's bar
    assert np.isnan(aligned["daily_return"].iloc[0])  # first equity date has no prior -> NaN


def test_window_universe_and_floor():
    dates = pd.bdate_range("2015-01-01", periods=400)
    rows = _equities(["SPY", "TLT"], dates)
    btc_dates = dates[-330:]
    rows += [
        {"symbol": "BTC", "date": d, "daily_return": 0.01, "asset_class": "crypto"}
        for d in btc_dates
    ]
    ad = pd.DataFrame(rows)
    aligned = compute.btc_aligned_returns(ad)
    floor = aligned.dropna(subset=["daily_return"])["date"].min()

    ex2002 = compute.window_asset_daily(ad, windows.EX_BTC_2002)
    assert set(ex2002["symbol"]) == {"SPY", "TLT"}  # BTC excluded
    assert ex2002["date"].min() == dates[0]  # full history, NOT truncated to BTC's start

    ex2015 = compute.window_asset_daily(
        ad, windows.EX_BTC_2015, btc_floor=floor, btc_aligned=aligned
    )
    assert set(ex2015["symbol"]) == {"SPY", "TLT"}  # still BTC-free
    assert ex2015["date"].min() == floor

    inc2015 = compute.window_asset_daily(
        ad, windows.INC_BTC_2015, btc_floor=floor, btc_aligned=aligned
    )
    assert set(inc2015["symbol"]) == {"SPY", "TLT", "BTC"}
    assert inc2015[inc2015["symbol"] == "BTC"]["date"].min() == floor
    # the same-period control: ex_btc_2015 and inc_btc_2015 share an identical equity date set
    assert set(ex2015["date"]) == set(inc2015[inc2015["symbol"] == "SPY"]["date"])


def test_ex_btc_2002_panel_not_collapsed_by_btc_presence():
    # The merged-panel trap: build_returns_panel(...).dropna(how='any') on a BTC-included panel
    # would collapse to BTC's start. The per-window ex_btc_2002 frame excludes BTC, so its panel
    # keeps the full non-crypto history.
    dates = pd.bdate_range("2015-01-01", periods=400)
    rows = _equities(["SPY", "TLT"], dates)
    rows += [
        {"symbol": "BTC", "date": d, "daily_return": 0.01, "asset_class": "crypto"}
        for d in dates[-330:]
    ]
    ad = pd.DataFrame(rows)
    panel = compute.build_returns_panel(compute.window_asset_daily(ad, windows.EX_BTC_2002))
    assert "BTC" not in panel.columns
    assert len(panel.dropna(how="any")) == 400  # full history survives, not truncated to 330


def _long_multi(non_crypto_days: int = 400, btc_days: int = 330) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=non_crypto_days)
    rng = np.random.default_rng(1)
    rows = []
    for sym in ("SPY", "TLT"):
        for d, r in zip(dates, rng.normal(0.0004, 0.01, non_crypto_days), strict=True):
            rows.append(
                {"symbol": sym, "date": d, "daily_return": float(r), "asset_class": "equities"}
            )
    for d, r in zip(dates[-btc_days:], rng.normal(0.0006, 0.02, btc_days), strict=True):
        rows.append({"symbol": "BTC", "date": d, "daily_return": float(r), "asset_class": "crypto"})
    return pd.DataFrame(rows)


def test_cmd_portfolio_lands_all_three_windows(monkeypatch, tmp_path):
    db = tmp_path / "p.duckdb"
    setup = duckdb.connect(str(db))
    setup.execute("create schema if not exists marts")
    setup.register("_fad", _long_multi())
    setup.execute("create table marts.fct_asset_daily as select * from _fad")
    setup.close()

    monkeypatch.setattr(cli, "connect", lambda *a, **k: duckdb.connect(str(db)))
    assert cli.cmd_portfolio(argparse.Namespace()) == 0

    check = duckdb.connect(str(db))
    try:
        wins = check.execute(
            "select distinct window_id from raw.portfolio_returns order by window_id"
        ).fetchall()
        # ex_btc_2002 spans the full history; ex_btc_2015 starts at the BTC floor (later).
        spans = check.execute(
            "select window_id, min(date) as lo from raw.portfolio_returns "
            "where window_id in ('ex_btc_2002', 'ex_btc_2015') group by window_id"
        ).df()
    finally:
        check.close()
    assert {w[0] for w in wins} == {"ex_btc_2002", "ex_btc_2015", "inc_btc_2015"}
    lo = dict(zip(spans["window_id"], spans["lo"], strict=True))
    assert lo["ex_btc_2002"] < lo["ex_btc_2015"]  # the 2002 window is genuinely longer


def test_window_restricts_to_universe_and_floors_to_common_start():
    """ex_btc_2002 keeps only PORTFOLIO_UNIVERSE sleeves (drops redundant QQQ + FX) and floors to
    the latest per-sleeve inception, so every rebalance optimises over the full universe."""
    early = pd.bdate_range("2000-01-03", periods=1600)  # long history that spans past GLD's start
    late = pd.bdate_range("2004-11-18", periods=300)  # a GLD-style later inception (overlaps early)
    rng = np.random.default_rng(2)
    rows = []
    for sym, dts, cls in [
        ("SPY", early, "equities"),
        ("QQQ", early, "equities"),  # redundant equity beta — must be dropped
        ("EURUSD", early, "fx"),  # FX — must be dropped
        ("TLT", early, "bonds"),
        ("GLD", late, "commodities"),  # later inception → sets the common-history floor
    ]:
        for d, r in zip(dts, rng.normal(0.0, 0.01, len(dts)), strict=True):
            rows.append({"symbol": sym, "date": d, "daily_return": float(r), "asset_class": cls})
    ad = pd.DataFrame(rows)

    ex2002 = compute.window_asset_daily(ad, windows.EX_BTC_2002)
    assert set(ex2002["symbol"]) == {"SPY", "TLT", "GLD"}  # universe only; QQQ + EURUSD dropped
    assert ex2002["date"].min() == late[0]  # floored to GLD's (latest) inception, not 2000
