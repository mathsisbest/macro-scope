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


def test_attribution_chart_uses_only_the_selected_strategy():
    attr = pd.DataFrame(
        {
            "strategy": ["equal_weight", "equal_weight", "sixty_forty"],
            "symbol": ["SPY", "TLT", "SPY"],
            "contribution_to_return": [0.02, -0.01, 0.05],
            "contribution_to_risk": [0.6, 0.4, 1.0],
        }
    )
    fig = charts.attribution_chart(attr, "equal_weight")
    bar = fig.data[0]
    assert set(bar.y) == {"SPY", "TLT"}  # only the selected strategy's assets, not sixty_forty's
    assert len(bar.x) == 2


def test_regime_sharpe_chart_one_trace_per_strategy_in_low_med_high_order():
    regime = pd.DataFrame(
        {
            "strategy": ["equal_weight"] * 3 + ["sixty_forty"] * 3,
            "regime": ["High", "Low", "Medium"] * 2,  # unordered on purpose
            "ann_return": [0.0] * 6,
            "ann_vol": [0.1] * 6,
            "ann_sharpe": [-0.3, -2.0, 1.1, 3.99, 1.66, 2.84],
        }
    )
    fig = charts.regime_sharpe_chart(regime)
    assert len(fig.data) == 2  # one grouped trace per strategy
    assert list(fig.data[0].x) == ["Low", "Medium", "High"]  # reindexed to a sensible order


def _gate(weight: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-28"]),
            "forecast_skill": [0.0, weight],
            "forecast_weight": [0.0, weight],
        }
    )


def _ml_pair(distinguishable: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_a": ["mvo_histmean"],
            "strategy_b": ["mvo_ml"],
            "sharpe_diff": [0.5 if distinguishable else 0.0],
            "diff_lo": [0.1 if distinguishable else -0.2],
            "diff_hi": [0.9 if distinguishable else 0.2],
            "distinguishable": [distinguishable],
        }
    )


def test_ml_verdict_reports_no_edge_when_not_distinguishable():
    verdict = charts.ml_verdict(_gate(0.0), _ml_pair(distinguishable=False))
    assert "0%" in verdict and "did not beat" in verdict


def test_ml_verdict_reports_edge_when_distinguishable():
    verdict = charts.ml_verdict(_gate(0.20), _ml_pair(distinguishable=True))
    assert "distinguishable" in verdict
    assert "did not beat" not in verdict


def _btc_effect(rows: list) -> pd.DataFrame:
    cols = [
        "strategy",
        "sharpe_ex",
        "sharpe_inc",
        "sharpe_diff",
        "diff_lo",
        "diff_hi",
        "distinguishable",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_btc_effect_verdict_reports_distinguishable_hurt():
    eff = _btc_effect(
        [
            ["equal_weight", 1.26, -0.32, -1.58, -2.76, -0.32, True],
            ["sixty_forty", 3.73, 3.73, 0.0, 0.0, 0.0, False],
        ]
    )
    verdict = charts.btc_effect_verdict(eff)
    assert "1 of 2" in verdict and "hurt" in verdict and "Equal weight" in verdict


def test_btc_effect_verdict_reports_none_when_all_within_noise():
    eff = _btc_effect([["mvo_ml", -0.6, -0.51, 0.09, -0.24, 0.37, False]])
    assert "no statistically distinguishable difference" in charts.btc_effect_verdict(eff)


def test_btc_effect_verdict_handles_empty():
    assert "not computed" in charts.btc_effect_verdict(_btc_effect([]))
