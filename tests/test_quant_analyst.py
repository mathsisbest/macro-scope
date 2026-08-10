"""The 'AI Quant Analyst' brief is a plain-English, deterministic post-mortem of the portfolio
marts.

Covers:
- assemble_quant_brief(): the pure template — scoreboard, winners/losers, significance verdicts
  (PR #154 return pairs + Sharpe pairs), regime context, attribution, ML gate, BTC effect.
- noise honesty: no distinguishable gap -> the brief calls the ranking noise, never skill.
- deterministic: same facts -> byte-identical output.
- degraded inputs: empty facts, missing sections, NaN/None values, a single strategy.
- generate_quant_brief(): template by default (no key); LLM rephrase optional; output validation
  + redaction mirror narrative.py (llm-rejected / llm-failed engine tags, secrets scrubbed).
- PortfolioFacts TypedDict key-set contract — an unexpected key raises ValueError.
"""

import logging

import pytest

from mmi.ai import quant_analyst

# ---------------------------------------------------------------------------
# Fact fixtures
# ---------------------------------------------------------------------------


def _strategy(s: str, sharpe: float) -> dict:
    return {
        "strategy": s,
        "sharpe": sharpe,
        "sharpe_lo": sharpe - 0.08,
        "sharpe_hi": sharpe + 0.08,
        "n_obs": 5124,
        "n_boot": 2000,
        "ci_pct": 0.9,
    }


def _rich_facts() -> dict:
    """Realistic PortfolioFacts: 8 strategies, pairs, regime, attribution, ML gate, BTC effect."""
    facts = {
        "as_of": "2026-08-10 12:00 UTC",
        "data_date": "2026-08-07",
        "window_id": "ex_btc_2002",
        "strategies": [
            _strategy("equal_weight", 0.62),
            _strategy("inverse_vol", 0.58),
            _strategy("risk_parity", 0.55),
            _strategy("sixty_forty", 0.51),
            _strategy("mvo_histmean", 0.42),
            _strategy("mvo_ml", 0.41),
            _strategy("ml_tilt", 0.38),
            _strategy("ml_regime", 0.35),
        ],
        "sharpe_pairs": [
            {
                "strategy_a": "equal_weight",
                "strategy_b": "sixty_forty",
                "sharpe_diff": 0.11,
                "diff_lo": -0.04,
                "diff_hi": 0.26,
                "distinguishable": False,
            },
            {
                "strategy_a": "inverse_vol",
                "strategy_b": "sixty_forty",
                "sharpe_diff": 0.07,
                "diff_lo": -0.08,
                "diff_hi": 0.22,
                "distinguishable": False,
            },
            {
                "strategy_a": "mvo_histmean",
                "strategy_b": "mvo_ml",
                "sharpe_diff": 0.01,
                "diff_lo": -0.10,
                "diff_hi": 0.12,
                "distinguishable": False,
            },
        ],
        "return_pairs": [
            {
                "strategy_a": "equal_weight",
                "strategy_b": "sixty_forty",
                "ann_return_diff": 0.021,
                "diff_lo": 0.004,
                "diff_hi": 0.038,
                "p_value": 0.015,
                "distinguishable": True,
            },
            {
                "strategy_a": "inverse_vol",
                "strategy_b": "sixty_forty",
                "ann_return_diff": 0.012,
                "diff_lo": -0.005,
                "diff_hi": 0.029,
                "p_value": 0.31,
                "distinguishable": False,
            },
        ],
        "regime": [
            {
                "strategy": "equal_weight",
                "regime": "Low",
                "n_days": 1400,
                "day_share": 0.35,
                "ann_return": 0.11,
                "ann_vol": 0.09,
                "ann_sharpe": 1.22,
            },
            {
                "strategy": "sixty_forty",
                "regime": "Low",
                "n_days": 1400,
                "day_share": 0.35,
                "ann_return": 0.10,
                "ann_vol": 0.09,
                "ann_sharpe": 1.10,
            },
            {
                "strategy": "risk_parity",
                "regime": "High",
                "n_days": 800,
                "day_share": 0.20,
                "ann_return": 0.04,
                "ann_vol": 0.13,
                "ann_sharpe": 0.31,
            },
            {
                "strategy": "equal_weight",
                "regime": "High",
                "n_days": 800,
                "day_share": 0.20,
                "ann_return": 0.02,
                "ann_vol": 0.16,
                "ann_sharpe": 0.13,
            },
        ],
        "attribution": [
            {
                "strategy": "equal_weight",
                "symbol": "SPY",
                "contribution_to_return": 0.38,
                "contribution_to_risk": 0.40,
            },
            {
                "strategy": "equal_weight",
                "symbol": "TLT",
                "contribution_to_return": 0.29,
                "contribution_to_risk": 0.33,
            },
            {
                "strategy": "equal_weight",
                "symbol": "GLD",
                "contribution_to_return": 0.15,
                "contribution_to_risk": 0.27,
            },
            {
                "strategy": "equal_weight",
                "symbol": "(costs)",
                "contribution_to_return": -0.012,
                "contribution_to_risk": 0.0,
            },
            {
                "strategy": "sixty_forty",
                "symbol": "SPY",
                "contribution_to_return": 0.33,
                "contribution_to_risk": 0.60,
            },
            {
                "strategy": "sixty_forty",
                "symbol": "(costs)",
                "contribution_to_return": -0.011,
                "contribution_to_risk": 0.0,
            },
        ],
        "btc_effect": [
            {
                "strategy": "equal_weight",
                "sharpe_ex": 0.40,
                "sharpe_inc": 0.38,
                "sharpe_diff": -0.02,
                "diff_lo": -0.09,
                "diff_hi": 0.05,
                "distinguishable": False,
            },
            {
                "strategy": "sixty_forty",
                "sharpe_ex": 0.45,
                "sharpe_inc": 0.45,
                "sharpe_diff": 0.00,
                "diff_lo": -0.07,
                "diff_hi": 0.07,
                "distinguishable": False,
            },
        ],
        "ml_gate": {"mean_forecast_weight": 0.04, "n_dates": 26},
    }
    return facts


def _facts_all_noise() -> dict:
    """Every Sharpe + return gap within noise — no distinguishable comparison anywhere."""
    facts = _rich_facts()
    for p in facts["return_pairs"]:
        p["distinguishable"] = False
        p["p_value"] = 0.55
    for p in facts["sharpe_pairs"]:
        p["distinguishable"] = False
    return facts


# ---------------------------------------------------------------------------
# assemble_quant_brief — structure & content
# ---------------------------------------------------------------------------


def test_brief_has_all_sections():
    text = quant_analyst.assemble_quant_brief(_rich_facts())
    for header in [
        "AI Quant Analyst — portfolio post-mortem",
        "**Scoreboard**",
        "**What won, what lost**",
        "**Is the edge real or noise?**",
        "**Regime context**",
        "**What drove returns**",
        "**ML experiment (mvo_ml)**",
        "**BTC impact**",
        "_Caveat:",
    ]:
        assert header in text
    assert "ex_btc_2002" in text or "~2004–present" in text
    assert "no LLM key set" in text


def test_brief_scoreboard_sorted_best_first_with_ci():
    text = quant_analyst.assemble_quant_brief(_rich_facts())
    ew = text.index("Equal weight: Sharpe 0.62 (90% CI 0.54–0.70)")
    inv = text.index("Inverse vol: Sharpe 0.58")
    reg = text.index("Risk parity: Sharpe 0.55")
    mlr = text.index("ML regime: Sharpe 0.35")
    assert ew < inv < reg < mlr


def test_brief_best_worst_benchmark_anchored():
    text = quant_analyst.assemble_quant_brief(_rich_facts())
    assert "Best risk-adjusted: Equal weight (Sharpe 0.62)" in text
    assert "vs 0.51 for the 60/40 benchmark" in text
    assert "Worst: ML regime (Sharpe 0.35)" in text


def test_brief_distinguishable_return_gap_is_named():
    text = quant_analyst.assemble_quant_brief(_rich_facts())
    assert "Return gaps: 1 of 2 annualised-return differences are distinguishable" in text
    assert "strongest evidence p ≤ 0.015" in text
    assert "Equal weight vs 60/40 benchmark" in text
    assert "The remaining gaps are within noise" in text


def test_brief_all_noise_is_called_noise_not_skill():
    text = quant_analyst.assemble_quant_brief(_facts_all_noise())
    assert "none of the 3 strategy comparisons is statistically distinguishable" in text
    assert "none of the 2 annualised-return differences is distinguishable" in text
    assert "the strategy ranking is not statistically separable from noise" in text
    assert "do not read skill into the winners/losers" in text
    assert "no reliable out-of-sample edge yet" in text


def test_brief_regime_context_names_leaders():
    text = quant_analyst.assemble_quant_brief(_rich_facts())
    assert "Low-vol: Equal weight leads (Sharpe 1.22); benchmark 1.10" in text
    assert "High-vol: Risk parity leads (Sharpe 0.31)" in text


def test_brief_attribution_top_contributors_and_costs():
    text = quant_analyst.assemble_quant_brief(_rich_facts())
    assert "Equal weight: SPY +38.0%, TLT +29.0%, GLD +15.0%, costs -1.20%" in text
    assert "60/40 benchmark: SPY +33.0%, costs -1.10%" in text


def test_brief_ml_gate_reports_no_edge_by_default():
    text = quant_analyst.assemble_quant_brief(_rich_facts())
    assert "mean weight of 4% in the blend over 26 rebalances" in text
    assert "mvo_ml is not statistically distinguishable from the historical-mean baseline" in text
    assert "no reliable out-of-sample edge yet" in text


def test_brief_ml_gate_distinguishable_edge():
    facts = _rich_facts()
    for p in facts["sharpe_pairs"]:
        if {p["strategy_a"], p["strategy_b"]} == {"mvo_histmean", "mvo_ml"}:
            p["distinguishable"] = True
    text = quant_analyst.assemble_quant_brief(facts)
    msg = "mvo_ml's Sharpe is statistically distinguishable from the historical-mean baseline"
    assert msg in text


def test_brief_btc_no_distinguishable_difference():
    text = quant_analyst.assemble_quant_brief(_rich_facts())
    assert "Adding BTC made no statistically distinguishable difference to any of the 2" in text


def test_brief_btc_helped_and_hurt():
    facts = _rich_facts()
    facts["btc_effect"][0]["distinguishable"] = True
    facts["btc_effect"][1]["sharpe_diff"] = 0.05
    facts["btc_effect"][1]["distinguishable"] = True
    text = quant_analyst.assemble_quant_brief(facts)
    assert "for 2 of 2 strategies" in text
    assert "hurt Equal weight (Δ-0.02)" in text
    assert "helped 60/40 benchmark (Δ+0.05)" in text


def test_brief_missing_btc_ml_sections_omitted():
    facts = _rich_facts()
    del facts["btc_effect"]
    del facts["ml_gate"]
    text = quant_analyst.assemble_quant_brief(facts)
    assert "**BTC impact**" not in text
    assert "**ML experiment (mvo_ml)**" not in text
    assert "**What won, what lost**" in text


# ---------------------------------------------------------------------------
# Degraded inputs — the brief must never crash and must stay honest
# ---------------------------------------------------------------------------


def test_brief_empty_facts_is_honest_placeholder():
    text = quant_analyst.assemble_quant_brief({"as_of": "x", "window_id": "ex_btc_2002"})
    assert "Portfolio marts are not available yet for this window" in text
    assert "run the backtest" in text


def test_brief_single_strategy_no_pairs():
    facts = {"as_of": "x", "data_date": "2026-08-07", "window_id": "ex_btc_2002"}
    facts["strategies"] = [_strategy("equal_weight", 0.62)]
    text = quant_analyst.assemble_quant_brief(facts)
    assert "Not enough strategy comparisons to test whether the gaps are real" in text
    assert "Best risk-adjusted: Equal weight (Sharpe 0.62)" in text


def test_brief_nan_and_none_values_do_not_crash():
    facts = _rich_facts()
    facts["strategies"][0]["sharpe"] = float("nan")
    facts["strategies"][0]["sharpe_lo"] = None
    facts["strategies"][0]["sharpe_hi"] = None
    facts["regime"][1]["ann_sharpe"] = None
    facts["attribution"][1]["contribution_to_return"] = None
    text = quant_analyst.assemble_quant_brief(facts)
    assert "Equal weight: Sharpe 0.00" in text
    assert "Low-vol: Equal weight leads" in text


def test_brief_deterministic_byte_identical():
    a = quant_analyst.assemble_quant_brief(_rich_facts())
    b = quant_analyst.assemble_quant_brief(_rich_facts())
    assert a == b


def test_brief_window_labels():
    for window_id, label in quant_analyst._WINDOW_LABELS.items():
        text = quant_analyst.assemble_quant_brief(
            {"as_of": "x", "data_date": "d", "window_id": window_id}
        )
        assert label in text


def test_brief_note_in_header():
    text = quant_analyst.assemble_quant_brief(_rich_facts(), note="LLM temporarily unavailable")
    assert "_(deterministic template — LLM temporarily unavailable)_" in text


# ---------------------------------------------------------------------------
# PortfolioFacts key-set contract
# ---------------------------------------------------------------------------


def test_validate_facts_keys_rejects_unexpected_key():
    with pytest.raises(ValueError, match="unexpected keys"):
        quant_analyst._validate_facts_keys({"as_of": "x", "UNKNOWN_KEY": 1})


def test_validate_facts_keys_accepts_all_valid_keys():
    quant_analyst._validate_facts_keys(_rich_facts())


# ---------------------------------------------------------------------------
# generate_quant_brief — template floor, LLM path, validation + redaction
# ---------------------------------------------------------------------------


def test_generate_quant_brief_offline_template_without_key(monkeypatch):
    monkeypatch.setattr(quant_analyst.llm, "available", lambda: False)
    text, engine = quant_analyst.generate_quant_brief(_rich_facts())
    assert "_(deterministic template — no LLM key set)_" in text
    assert engine == "offline-template"


def test_generate_quant_brief_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr(quant_analyst.llm, "available", lambda: True)
    monkeypatch.setattr(
        quant_analyst.llm, "complete", lambda *_a, **_k: ("Grounded rewrite.", "mock:test")
    )
    text, engine = quant_analyst.generate_quant_brief(_rich_facts())
    assert text == "Grounded rewrite."
    assert engine == "mock:test"


def test_generate_quant_brief_llm_failure_falls_back_and_redacts(monkeypatch, caplog):
    leaked = (
        "503 Server Error for url "
        "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key=AIzaSECRET123"
    )
    monkeypatch.setattr(quant_analyst.llm, "available", lambda: True)

    def _boom(*_a, **_k):
        raise RuntimeError(leaked)

    monkeypatch.setattr(quant_analyst.llm, "complete", _boom)
    with caplog.at_level(logging.WARNING):
        text, engine = quant_analyst.generate_quant_brief(_rich_facts())
    assert "_(deterministic template — LLM temporarily unavailable)_" in text
    assert engine == "offline-template (llm-failed)"
    assert "AIzaSECRET123" not in caplog.text
    assert "key=***" in caplog.text


def test_generate_quant_brief_rejects_key_shaped_llm_output(monkeypatch):
    monkeypatch.setattr(quant_analyst.llm, "available", lambda: True)
    monkeypatch.setattr(
        quant_analyst.llm,
        "complete",
        lambda *_a, **_k: ("Summary. api_key=SUPER_SECRET_XYZ here.", "mock:test"),
    )
    text, engine = quant_analyst.generate_quant_brief(_rich_facts())
    assert "_(deterministic template — LLM output failed validation)_" in text
    assert engine == "offline-template (llm-rejected)"


def test_generate_quant_brief_redacts_returned_body(monkeypatch):
    monkeypatch.setattr(quant_analyst.llm, "available", lambda: False)
    tainted = "Grounded. bearer FAKETOKEN123 Rewrite."
    monkeypatch.setattr(quant_analyst, "assemble_quant_brief", lambda *_a, **_k: tainted)
    text, _ = quant_analyst.generate_quant_brief(_rich_facts())
    assert "FAKETOKEN123" not in text
    assert "bearer ***" in text


def test_generate_quant_brief_rejects_unknown_keys(monkeypatch):
    monkeypatch.setattr(quant_analyst.llm, "available", lambda: False)
    with pytest.raises(ValueError, match="unexpected keys"):
        quant_analyst.generate_quant_brief({"as_of": "x", "EXTRA": 1})
