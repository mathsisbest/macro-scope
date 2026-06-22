"""The bootstrap scorecard/verdict builders summarise the pairwise result honestly."""

import pandas as pd
from dashboard.components import charts

_COLS = ["strategy_a", "strategy_b", "sharpe_diff", "diff_lo", "diff_hi", "distinguishable"]


def _pairs(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLS)


def test_verdict_when_nothing_is_distinguishable():
    pairs = _pairs(
        [
            ["equal_weight", "risk_parity", 0.0, -0.1, 0.1, False],
            ["equal_weight", "inverse_vol", -0.05, -0.2, 0.1, False],
        ]
    )
    verdict = charts.distinguishability_verdict(pairs)
    assert "None" in verdict and "within noise" in verdict


def test_verdict_lists_only_distinguishable_pairs_with_labels():
    pairs = _pairs(
        [
            ["equal_weight", "sixty_forty", -3.2, -5.2, -1.3, True],
            ["inverse_vol", "risk_parity", 0.06, -0.08, 0.18, False],
        ]
    )
    verdict = charts.distinguishability_verdict(pairs)
    assert "1 of 2" in verdict
    assert "Equal weight vs 60/40 benchmark" in verdict  # labelled, not raw keys
    assert "Risk parity" not in verdict  # the indistinguishable pair is omitted


def test_verdict_handles_empty():
    assert "Not enough" in charts.distinguishability_verdict(_pairs([]))


def test_scorecard_shape_and_labels():
    stats = pd.DataFrame(
        {
            "strategy": ["sixty_forty", "equal_weight"],
            "sharpe": [2.5, -0.6],
            "sharpe_lo": [0.4, -2.7],
            "sharpe_hi": [4.6, 1.6],
            "n_obs": [147, 147],
            "n_boot": [2000, 2000],
            "ci_pct": [0.9, 0.9],
        }
    )
    sc = charts.portfolio_scorecard(stats)
    assert list(sc.columns) == ["Sharpe", "CI low", "CI high"]
    assert "60/40 benchmark" in sc.index  # raw key mapped to a display label
