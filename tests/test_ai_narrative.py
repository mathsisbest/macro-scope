"""The market brief is numerically grounded: portfolio stats come from the mart, not invented."""

import duckdb
import pandas as pd

from mmi.ai import narrative


def _con_with_portfolio() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    rows = pd.DataFrame(
        {
            "strategy": ["risk_parity", "risk_parity", "sixty_forty", "sixty_forty"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            "daily_return": [0.0, 0.02, 0.0, 0.01],
            "cumulative_return": [0.0, 0.05, 0.0, 0.08],
            "drawdown": [0.0, -0.03, 0.0, -0.01],
            "rolling_sharpe_252": [None, 1.40, None, 1.90],
        }
    )
    con.register("_r", rows)
    con.execute("create table marts.fct_portfolio_returns as select * from _r")
    con.unregister("_r")
    return con


def test_gather_facts_includes_portfolio_stats():
    con = _con_with_portfolio()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    by = {p["strategy"]: p for p in facts["portfolio"]}
    assert set(by) == {"risk_parity", "sixty_forty"}
    assert by["sixty_forty"]["total_return"] == 0.08  # last cumulative_return, not the peak
    assert by["sixty_forty"]["max_drawdown"] == -0.01
    assert by["risk_parity"]["sharpe"] == 1.40  # latest non-null rolling Sharpe


def test_offline_brief_grounds_strategy_numbers():
    facts = {
        "as_of": "2020-01-02 00:00 UTC",
        "portfolio": [
            {
                "strategy": "sixty_forty",
                "total_return": 0.08,
                "max_drawdown": -0.01,
                "ann_vol": 0.12,
                "sharpe": 1.90,
            },
        ],
    }
    brief = narrative._offline_brief(facts)
    assert "60/40 benchmark" in brief
    assert "+8.0% total return" in brief  # rendered from the fact, not fabricated
    assert "max drawdown -1.0%" in brief
    assert "Sharpe 1.90" in brief


def test_offline_brief_handles_missing_sharpe_and_no_portfolio():
    # No portfolio facts -> no strategy section, brief still renders.
    assert "Strategy comparison" not in narrative._offline_brief({"as_of": "x"})
    # Null Sharpe -> 'n/a', never a crash.
    facts = {
        "as_of": "x",
        "portfolio": [
            {"strategy": "equal_weight", "total_return": 0.0, "max_drawdown": 0.0, "sharpe": None}
        ],
    }
    assert "Sharpe n/a" in narrative._offline_brief(facts)
