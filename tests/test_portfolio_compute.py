"""btc_aligned_returns: interior-NaN warning and alignment behaviour."""

import logging

import numpy as np
import pandas as pd

from mmi.portfolio import compute

BTC_LOGGER = "portfolio.compute"


def _frame(btc_dates: list[str], btc_returns: list[float]) -> pd.DataFrame:
    eq_dates = pd.bdate_range("2020-01-02", periods=5)
    rows = [
        {"symbol": "SPY", "date": d, "daily_return": 0.01, "asset_class": "equities"}
        for d in eq_dates
    ]
    rows += [
        {"symbol": "BTC", "date": pd.Timestamp(d), "daily_return": r, "asset_class": "crypto"}
        for d, r in zip(btc_dates, btc_returns, strict=True)
    ]
    return pd.DataFrame(rows)


def test_btc_aligned_returns_warns_on_interior_nan(caplog):
    df = _frame(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"],
        [0.10, 0.10, np.nan, 0.10, 0.10],
    )
    with caplog.at_level(logging.WARNING, logger=BTC_LOGGER):
        aligned = compute.btc_aligned_returns(df)
    assert [r for r in caplog.records if r.name == BTC_LOGGER]
    assert "interior NaN observation(s)" in caplog.text
    assert "1 interior NaN" in caplog.text
    assert "0/0" in caplog.text  # no leading/trailing gaps

    filled = _frame(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"],
        [0.10, 0.10, 0.0, 0.10, 0.10],
    )
    expected = compute.btc_aligned_returns(filled)
    pd.testing.assert_frame_equal(aligned, expected)


def test_btc_aligned_returns_silent_on_leading_trailing_nan(caplog):
    df = _frame(
        ["2020-01-01", "2020-01-03", "2020-01-04", "2020-01-07", "2020-01-08"],
        [np.nan, 0.10, 0.10, 0.10, np.nan],
    )
    with caplog.at_level(logging.WARNING, logger=BTC_LOGGER):
        aligned = compute.btc_aligned_returns(df)
    assert "interior NaN" not in caplog.text
    assert list(aligned.columns) == ["date", "daily_return"]
    assert aligned["date"].is_monotonic_increasing


def test_btc_aligned_returns_empty_btc_returns_empty_frame():
    df = _frame(["2020-01-02"], [0.10])
    df = df[df["symbol"] != "BTC"]
    aligned = compute.btc_aligned_returns(df)
    assert aligned.empty
    assert list(aligned.columns) == ["date", "daily_return"]
