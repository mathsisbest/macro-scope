"""The market brief is numerically grounded: portfolio stats + CIs come from the marts, and the
brief hedges (says "not distinguishable") when the bootstrap says so — never invented."""

import duckdb
import pandas as pd

from mmi.ai import narrative


def _con_with_portfolio() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    returns = pd.DataFrame(
        {
            "strategy": ["risk_parity", "risk_parity", "sixty_forty", "sixty_forty"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            "daily_return": [0.0, 0.02, 0.0, 0.01],
            "cumulative_return": [0.0, 0.05, 0.0, 0.08],
            "drawdown": [0.0, -0.03, 0.0, -0.01],
            "rolling_sharpe_252": [None, 1.40, None, 1.90],
        }
    )
    stats = pd.DataFrame(
        {
            "strategy": ["risk_parity", "sixty_forty"],
            "sharpe": [0.30, 1.10],
            "sharpe_lo": [-0.50, 0.20],
            "sharpe_hi": [1.10, 2.00],
            "n_obs": [2, 2],
            "n_boot": [100, 100],
            "ci_pct": [0.9, 0.9],
        }
    )
    pairs = pd.DataFrame(
        {
            "strategy_a": ["risk_parity"],
            "strategy_b": ["sixty_forty"],
            "sharpe_a": [0.30],
            "sharpe_b": [1.10],
            "sharpe_diff": [-0.80],
            "diff_lo": [-1.5],
            "diff_hi": [0.1],
            "distinguishable": [False],
        }
    )
    for name, df in [
        ("fct_portfolio_returns", returns),
        ("fct_portfolio_strategy_stats", stats),
        ("fct_portfolio_strategy_pairs", pairs),
    ]:
        con.register("_t", df)
        con.execute(f"create table marts.{name} as select * from _t")
        con.unregister("_t")
    return con


def test_gather_facts_joins_returns_with_bootstrap_ci_and_pairs():
    con = _con_with_portfolio()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    by = {p["strategy"]: p for p in facts["portfolio"]}
    assert set(by) == {"risk_parity", "sixty_forty"}
    assert by["sixty_forty"]["total_return"] == 0.08  # returns-derived (last cumulative)
    assert by["sixty_forty"]["max_drawdown"] == -0.01
    assert by["sixty_forty"]["sharpe"] == 1.10  # bootstrap full-sample, not the rolling 1.90
    assert by["sixty_forty"]["sharpe_lo"] == 0.20
    assert by["sixty_forty"]["sharpe_hi"] == 2.00
    assert not facts["portfolio_pairs"][0]["distinguishable"]


def test_offline_brief_renders_sharpe_ci_and_hedges():
    facts = {
        "as_of": "2020-01-02 00:00 UTC",
        "portfolio": [
            {
                "strategy": "sixty_forty",
                "total_return": 0.08,
                "max_drawdown": -0.01,
                "sharpe": 1.10,
                "sharpe_lo": 0.20,
                "sharpe_hi": 2.00,
            },
        ],
        "portfolio_pairs": [
            {"strategy_a": "risk_parity", "strategy_b": "sixty_forty", "distinguishable": False},
        ],
    }
    brief = narrative._offline_brief(facts)
    assert "+8.0% total return" in brief
    assert "Sharpe 1.10 [0.20, 2.00]" in brief  # CI rendered from facts, not invented
    assert "no pair of strategies is distinguishable" in brief  # honest hedge


def test_offline_brief_lists_distinguishable_pairs_when_present():
    facts = {
        "as_of": "x",
        "portfolio": [
            {
                "strategy": "equal_weight",
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.1,
                "sharpe_lo": -0.1,
                "sharpe_hi": 0.3,
            }
        ],
        "portfolio_pairs": [
            {"strategy_a": "equal_weight", "strategy_b": "sixty_forty", "distinguishable": True},
        ],
    }
    brief = narrative._offline_brief(facts)
    assert "Equal weight vs 60/40 benchmark differ beyond bootstrap noise" in brief


def test_offline_brief_handles_no_portfolio_and_missing_sharpe():
    assert "Strategy comparison" not in narrative._offline_brief({"as_of": "x"})
    facts = {
        "as_of": "x",
        "portfolio": [
            {"strategy": "equal_weight", "total_return": 0.0, "max_drawdown": 0.0, "sharpe": None}
        ],
    }
    assert "Sharpe n/a" in narrative._offline_brief(facts)
