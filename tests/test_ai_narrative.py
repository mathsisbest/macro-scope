"""The market brief is numerically grounded: portfolio stats + CIs come from the marts, and the
brief hedges (says "not distinguishable") when the bootstrap says so — never invented."""

import duckdb
import pandas as pd

from mmi.ai import narrative


def _con_with_portfolio() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    # Every portfolio mart carries window_id; a second window (inc_btc_2015) with DELIBERATELY
    # different numbers proves gather_facts scopes to the brief's default window (ex_btc_2002).
    returns = pd.DataFrame(
        {
            "window_id": ["ex_btc_2002"] * 4 + ["inc_btc_2015"] * 2,
            "strategy": ["risk_parity", "risk_parity", "sixty_forty", "sixty_forty"]
            + ["sixty_forty", "sixty_forty"],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]
            ),
            "daily_return": [0.0, 0.02, 0.0, 0.01, 0.0, 0.5],
            "cumulative_return": [0.0, 0.05, 0.0, 0.08, 0.0, 0.99],
            "drawdown": [0.0, -0.03, 0.0, -0.01, 0.0, -0.40],
            "rolling_sharpe_252": [None, 1.40, None, 1.90, None, 9.0],
        }
    )
    stats = pd.DataFrame(
        {
            "window_id": ["ex_btc_2002", "ex_btc_2002", "inc_btc_2015"],
            "strategy": ["risk_parity", "sixty_forty", "sixty_forty"],
            "sharpe": [0.30, 1.10, 9.99],
            "sharpe_lo": [-0.50, 0.20, 9.0],
            "sharpe_hi": [1.10, 2.00, 11.0],
            "n_obs": [2, 2, 2],
            "n_boot": [100, 100, 100],
            "ci_pct": [0.9, 0.9, 0.9],
        }
    )
    pairs = pd.DataFrame(
        {
            "window_id": ["ex_btc_2002"],
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
    gate = pd.DataFrame(
        {
            "window_id": ["ex_btc_2002", "ex_btc_2002", "inc_btc_2015"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_skill": [0.0, 0.04, 0.8],
            "forecast_weight": [0.0, 0.02, 0.40],  # ex_btc_2002 mean = 0.01; inc = 0.40
        }
    )
    for name, df in [
        ("fct_portfolio_returns", returns),
        ("fct_portfolio_strategy_stats", stats),
        ("fct_portfolio_strategy_pairs", pairs),
        ("fct_portfolio_ml_gate", gate),
    ]:
        con.register("_t", df)
        con.execute(f"create table marts.{name} as select * from _t")
        con.unregister("_t")
    return con


def test_gather_facts_scopes_to_one_window_and_joins_ci_pairs_gate():
    con = _con_with_portfolio()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    by = {p["strategy"]: p for p in facts["portfolio"]}
    assert set(by) == {"risk_parity", "sixty_forty"}
    # the inc_btc_2015 rows (total_return 0.99, sharpe 9.99) must NOT leak in
    assert by["sixty_forty"]["total_return"] == 0.08  # ex_btc_2002 last cumulative, not 0.99
    assert by["sixty_forty"]["max_drawdown"] == -0.01  # not the inc_btc_2015 -0.40
    assert by["sixty_forty"]["sharpe"] == 1.10  # ex_btc_2002 bootstrap Sharpe, not 9.99
    assert by["sixty_forty"]["sharpe_lo"] == 0.20
    assert not facts["portfolio_pairs"][0]["distinguishable"]
    assert abs(facts["ml_gate"]["mean_weight"] - 0.01) < 1e-9  # ex_btc_2002 mean, not 0.40


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


def test_offline_brief_grounds_the_ml_gate_when_present():
    facts = {
        "as_of": "x",
        "portfolio": [
            {
                "strategy": "mvo_ml",
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.1,
                "sharpe_lo": -0.1,
                "sharpe_hi": 0.3,
            }
        ],
        "ml_gate": {"mean_weight": 0.01, "max_weight": 0.02},
    }
    brief = narrative._offline_brief(facts)
    assert "earned a mean weight of 1%" in brief
    assert "no reliable out-of-sample edge" in brief  # honest: ~0 weight -> matched the baseline


def test_offline_brief_omits_ml_gate_when_absent():
    facts = {
        "as_of": "x",
        "portfolio": [
            {"strategy": "mvo_ml", "total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.1}
        ],
    }
    assert "ML gate" not in narrative._offline_brief(facts)
