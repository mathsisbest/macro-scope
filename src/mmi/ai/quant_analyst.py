"""Plain-English 'AI Quant Analyst' brief for the Portfolio tab, assembled from the portfolio marts.

The brief is a structured post-mortem of the walk-forward backtest: what won/lost (scoreboard +
best/worst), whether the gaps are real or noise (the paired block-bootstrap significance verdicts),
regime context (performance conditioned on market volatility), what drove returns (attribution),
the ML experiment's verdict, and the BTC effect. It is deliberately honest: rankings that are not
statistically distinguishable are called noise, not skill.

Like narrative.py, it falls back to a deterministic template when no LLM key is configured (or the
call fails / the output is rejected), so the feature always works, CI stays hermetic, and the cost
stays at £0. The LLM is a pure rephrase pass: the deterministic template is the contract, and the
LLM only re-writes it in a more conversational register with the figures unchanged. Nothing is
persisted — the dashboard renders the brief from the marts at render time, so the ``market_brief``
schema is untouched.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from mmi.ai import llm
from mmi.ai.narrative import _validate_llm_output
from mmi.utils.logging import get_logger
from mmi.utils.redact import redact

log = get_logger("ai.quant_analyst")

# ---------------------------------------------------------------------------
# Facts TypedDict — contract-frozen key set.
#
# The facts dict consumed by assemble_quant_brief() must use EXACTLY these keys
# (some values are absent when a mart is missing, but no extra or missing keys).
# ---------------------------------------------------------------------------


class PortfolioFacts(TypedDict, total=False):
    """Typed contract for the facts dict assembled from the portfolio marts.

    All keys are optional (total=False) because individual marts may be absent when running
    against an empty DB (CI, seed, demo). 'as_of'/'data_date'/'window_id' are always produced;
    the rest are conditional on mart availability.
    """

    as_of: str
    data_date: str
    window_id: str
    strategies: list[dict[str, Any]]
    sharpe_pairs: list[dict[str, Any]]
    return_pairs: list[dict[str, Any]]
    regime: list[dict[str, Any]]
    attribution: list[dict[str, Any]]
    btc_effect: list[dict[str, Any]]
    ml_gate: dict[str, Any]


_FACTS_REQUIRED_KEYS: frozenset[str] = frozenset(PortfolioFacts.__annotations__)


def _validate_facts_keys(facts: dict) -> None:
    """Raise ValueError if facts contains unknown keys (extra/undocumented structure)."""
    extra = set(facts.keys()) - _FACTS_REQUIRED_KEYS
    if extra:
        raise ValueError(
            f"facts produced unexpected keys not in the PortfolioFacts contract: {extra!r}. "
            "Add them to PortfolioFacts or remove them from the caller."
        )


# ---------------------------------------------------------------------------
# Display labels for the brief (the dashboard's tab chart labels live in charts.py).
# ---------------------------------------------------------------------------

_WINDOW_LABELS: dict[str, str] = {
    "ex_btc_2002": "~2004–present · ex-BTC",
    "ex_btc_2015": "2015–present · ex-BTC (BTC era)",
    "inc_btc_2015": "2015–present · incl. BTC",
}

_STRATEGY_LABELS: dict[str, str] = {
    "equal_weight": "Equal weight",
    "inverse_vol": "Inverse vol",
    "risk_parity": "Risk parity",
    "sixty_forty": "60/40 benchmark",
    "mvo_histmean": "MVO (historical mean)",
    "mvo_ml": "MVO (ML forecast)",
    "ml_tilt": "ML tilt",
    "ml_regime": "ML regime",
}


def _n(x: object, default: float = 0.0) -> float:
    """Coerce a possibly-None/NaN numeric fact to a plain float (NaN is truthy, so guard it)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    return float(x)  # type: ignore[arg-type]


def _label(strategy: str) -> str:
    return _STRATEGY_LABELS.get(strategy, strategy)


# ---------------------------------------------------------------------------
# Section builders — each renders one section of the brief from its facts.
# ---------------------------------------------------------------------------


def _scoreboard_lines(strategies: list[dict[str, Any]]) -> list[str]:
    """One bullet per strategy: annualised Sharpe with its bootstrap CI, best first."""
    lines: list[str] = []
    for s in sorted(strategies, key=lambda r: _n(r.get("sharpe")), reverse=True):
        lo, hi = s.get("sharpe_lo"), s.get("sharpe_hi")
        has_ci = lo is not None and hi is not None and not pd.isna(lo) and not pd.isna(hi)
        ci_label = ""
        if has_ci:
            ci_label = f" ({_n(s.get('ci_pct', 0.9)) * 100:.0f}% CI {_n(lo):.2f}–{_n(hi):.2f})"
        name = _label(str(s.get("strategy", "")))
        lines.append(f"- {name}: Sharpe {_n(s.get('sharpe')):.2f}{ci_label}")
    return lines


def _best_worst_lines(strategies: list[dict[str, Any]]) -> list[str]:
    """What won and lost: the best/worst strategies by Sharpe, benchmark-anchored."""
    ranked = sorted(strategies, key=lambda r: _n(r.get("sharpe")), reverse=True)
    if not ranked:
        return []
    best, worst = ranked[0], ranked[-1]
    bench = next((r for r in strategies if r.get("strategy") == "sixty_forty"), None)
    best_name = _label(str(best.get("strategy", "")))
    lines: list[str] = []
    if bench is not None and best is not bench:
        bench_sharpe = _n(bench.get("sharpe"))
        lines.append(
            f"Best risk-adjusted: {best_name} (Sharpe {_n(best.get('sharpe')):.2f}) — "
            f"vs {_n(bench_sharpe):.2f} for the 60/40 benchmark."
        )
    else:
        lines.append(f"Best risk-adjusted: {best_name} (Sharpe {_n(best.get('sharpe')):.2f}).")
    if len(ranked) > 1:
        lines.append(
            f"Worst: {_label(str(worst.get('strategy', '')))} "
            f"(Sharpe {_n(worst.get('sharpe')):.2f})."
        )
    return lines


def _noise_lines(
    sharpe_pairs: list[dict[str, Any]], return_pairs: list[dict[str, Any]]
) -> list[str]:
    """Is the edge real or noise? Bootstrap distinguishability verdicts for both gaps."""
    lines: list[str] = []
    if not sharpe_pairs and not return_pairs:
        return ["Not enough strategy comparisons to test whether the gaps are real."]

    if sharpe_pairs:
        distinct = [p for p in sharpe_pairs if p.get("distinguishable")]
        n = len(sharpe_pairs)
        if distinct:
            named = "; ".join(
                f"{_label(str(p.get('strategy_a', '')))} vs {_label(str(p.get('strategy_b', '')))}"
                for p in distinct
            )
            lines.append(
                f"Sharpe gaps: {len(distinct)} of {n} comparisons are distinguishable ({named}); "
                "the rest are within noise."
            )
        else:
            lines.append(
                f"Sharpe gaps: none of the {n} strategy comparisons is statistically "
                "distinguishable — every difference CI includes zero, so the ranking is within "
                "noise at this sample size."
            )

    if return_pairs:
        distinct = [p for p in return_pairs if p.get("distinguishable")]
        n = len(return_pairs)
        if distinct:
            min_p = min(_n(p.get("p_value"), default=1.0) for p in distinct)
            named = "; ".join(
                f"{_label(str(p.get('strategy_a', '')))} vs {_label(str(p.get('strategy_b', '')))}"
                for p in distinct
            )
            lines.append(
                f"Return gaps: {len(distinct)} of {n} annualised-return differences are "
                f"distinguishable (strongest evidence p ≤ {min_p:.3g}): {named}. The remaining "
                "gaps are within noise."
            )
        else:
            lines.append(
                f"Return gaps: none of the {n} annualised-return differences is distinguishable "
                "— every bootstrap difference CI includes zero, i.e. the return gaps are within "
                "noise at this sample size."
            )

    if sharpe_pairs and return_pairs:
        all_noise = not any(p.get("distinguishable") for p in sharpe_pairs) and not any(
            p.get("distinguishable") for p in return_pairs
        )
        if all_noise:
            lines.append(
                "Bottom line: the strategy ranking is not statistically separable from noise — "
                "do not read skill into the winners/losers at this sample size."
            )
    return lines


def _regime_lines(regime: list[dict[str, Any]]) -> list[str]:
    """Per market-volatility regime: which strategy led on Sharpe there (benchmark-anchored)."""
    order = ["Low", "Medium", "High"]
    lines: list[str] = []
    for rv in order:
        rows = [
            r
            for r in regime
            if r.get("regime") == rv
            and r.get("ann_sharpe") is not None
            and not pd.isna(r.get("ann_sharpe"))
        ]
        if not rows:
            continue
        best = max(rows, key=lambda r: _n(r.get("ann_sharpe")))
        bench = next((r for r in rows if r.get("strategy") == "sixty_forty"), None)
        bench_txt = f"; benchmark {_n(bench.get('ann_sharpe')):.2f}" if bench else ""
        lines.append(
            f"- {rv}-vol: {_label(str(best.get('strategy', '')))} leads "
            f"(Sharpe {_n(best.get('ann_sharpe')):.2f}){bench_txt}."
        )
    return lines


def _attribution_lines(attribution: list[dict[str, Any]]) -> list[str]:
    """Per strategy: the biggest return contributors (top 3 assets) + the cost drag."""
    lines: list[str] = []
    for strategy in sorted({str(r["strategy"]) for r in attribution}):
        rows = [r for r in attribution if str(r["strategy"]) == strategy]
        costs = next((r for r in rows if str(r.get("symbol", "")) == "(costs)"), None)
        assets = sorted(
            (r for r in rows if str(r.get("symbol", "")) != "(costs)"),
            key=lambda r: _n(r.get("contribution_to_return")),
            reverse=True,
        )
        parts = [
            f"{r.get('symbol')} {_n(r.get('contribution_to_return')):+.1%}" for r in assets[:3]
        ]
        if costs:
            parts.append(f"costs {_n(costs.get('contribution_to_return')):+.2%}")
        if not parts:
            continue
        lines.append(f"- {_label(strategy)}: " + ", ".join(parts))
    return lines


def _ml_lines(ml_gate: dict[str, Any], sharpe_pairs: list[dict[str, Any]]) -> list[str]:
    """The ML experiment verdict: mean forecast weight in the blend + distinguishability."""
    if not ml_gate:
        return []
    mean_w = _n(ml_gate.get("mean_forecast_weight"))
    n_dates = ml_gate.get("n_dates")
    distinguishable = False
    if sharpe_pairs:
        match = [
            p
            for p in sharpe_pairs
            if {p.get("strategy_a"), p.get("strategy_b")} == {"mvo_histmean", "mvo_ml"}
        ]
        if match:
            distinguishable = bool(match[0].get("distinguishable"))
    rebalances = f" over {int(_n(n_dates)):,} rebalances" if n_dates else ""
    if distinguishable:
        return [
            f"The ML forecast earned a mean weight of {mean_w:.0%} in the blend{rebalances}, and "
            "mvo_ml's Sharpe is statistically distinguishable from the historical-mean baseline "
            "— the forecast is adding measurable value."
        ]
    return [
        f"The ML forecast earned a mean weight of {mean_w:.0%} in the blend{rebalances}; mvo_ml "
        "is not statistically distinguishable from the historical-mean baseline — no reliable "
        "out-of-sample edge yet, so the ML did not beat the simpler approach."
    ]


def _btc_lines(btc_effect: list[dict[str, Any]]) -> list[str]:
    """BTC effect verdict: how many strategies saw a distinguishable paired difference."""
    if not btc_effect:
        return []
    n = len(btc_effect)
    distinct = [r for r in btc_effect if r.get("distinguishable")]
    if not distinct:
        return [
            f"Adding BTC made no statistically distinguishable difference to any of the {n} "
            "strategies' Sharpe (same-period paired comparison, 2015 window)."
        ]
    hurt = [r for r in distinct if _n(r.get("sharpe_diff")) < 0]
    helped = [r for r in distinct if _n(r.get("sharpe_diff")) > 0]
    parts: list[str] = []
    if hurt:
        parts.append(
            "hurt "
            + ", ".join(
                f"{_label(str(r.get('strategy', '')))} (Δ{_n(r.get('sharpe_diff')):+.2f})"
                for r in hurt
            )
        )
    if helped:
        parts.append(
            "helped "
            + ", ".join(
                f"{_label(str(r.get('strategy', '')))} (Δ{_n(r.get('sharpe_diff')):+.2f})"
                for r in helped
            )
        )
    return [
        f"Adding BTC made a statistically distinguishable difference for {len(distinct)} of {n} "
        f"strategies: {'; '.join(parts)}."
    ]


# ---------------------------------------------------------------------------
# The deterministic template — the offline floor and the LLM prompt's fact block.
# ---------------------------------------------------------------------------


def assemble_quant_brief(facts: PortfolioFacts, note: str = "no LLM key set") -> str:
    """Assemble the deterministic plain-English brief from the portfolio-mart facts.

    Same facts → byte-identical output (pure). Reports the scoreboard, winners/losers,
    significance verdicts, regime context, attribution, ML gate and BTC effect honestly, without
    editorialising beyond what the bootstrap CIs support.
    """
    window_id = facts.get("window_id", "")
    data_date = facts.get("data_date", facts.get("as_of", ""))
    window_label = _WINDOW_LABELS.get(window_id, window_id) if window_id else window_id

    lines = [
        f"**AI Quant Analyst — portfolio post-mortem** _(deterministic template — {note})_",
        "",
        f"Backtest window: {window_label} · data as of {data_date}",
        "",
    ]

    strategies = facts.get("strategies", [])
    if not strategies:
        lines.append(
            "Portfolio marts are not available yet for this window — run the backtest "
            "(`mmi portfolio`) and rebuild the marts first."
        )
        return "\n".join(lines)

    scoreboard = _scoreboard_lines(strategies)
    lines += ["**Scoreboard** (annualised Sharpe with bootstrap CI)", *scoreboard, ""]

    won_lost = _best_worst_lines(strategies)
    if won_lost:
        lines += ["**What won, what lost**", *won_lost, ""]

    noise = _noise_lines(facts.get("sharpe_pairs", []), facts.get("return_pairs", []))
    if noise:
        lines += ["**Is the edge real or noise?**", *noise, ""]

    regime_lines = _regime_lines(facts.get("regime", []))
    if regime_lines:
        lines += [
            "**Regime context** (strategy Sharpe by market volatility regime)",
            *regime_lines,
            "",
        ]

    attr_lines = _attribution_lines(facts.get("attribution", []))
    if attr_lines:
        lines += ["**What drove returns** (attribution, full window)", *attr_lines, ""]

    ml_lines = _ml_lines(facts.get("ml_gate", {}), facts.get("sharpe_pairs", []))
    if ml_lines:
        lines += ["**ML experiment (mvo_ml)**", *ml_lines, ""]

    btc_lines = _btc_lines(facts.get("btc_effect", []))
    if btc_lines:
        lines += ["**BTC impact**", *btc_lines, ""]

    lines.append(
        "_Caveat: the bootstrap CIs quantify sampling noise; rankings that are not "
        "distinguishable are not evidence of skill. Assembled deterministically from the "
        "portfolio marts._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional LLM rephrase pass — provider-agnostic via mmi.ai.llm, template as the floor.
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a rigorous quantitative analyst writing the 'AI Quant Analyst' post-mortem for a "
    "walk-forward portfolio backtest, read by a sophisticated investor. Write structured, "
    "plain-English markdown.\n\n"
    "Keep the same sections and section order as the fact block: scoreboard, what won/lost, is "
    "the edge real or noise, regime context, what drove returns, ML experiment, BTC impact. "
    "Use ONLY the figures provided and quote them exactly as written — do not invent, estimate, "
    "or add precision. Frame causation cautiously.\n\n"
    "Be brutally honest about noise: if the fact block says a comparison is not distinguishable, "
    "say the gap is within noise and never imply skill. Keep it under 8000 characters. Tone: "
    "direct, opinionated but grounded, no hype, no filler."
)


def _build_prompt(facts: PortfolioFacts) -> str:
    """The deterministic template doubles as the LLM's fact block — grounded by construction."""
    return (
        "Here is the deterministic portfolio post-mortem assembled from the marts (every figure "
        "is real and pre-formatted). Rewrite it in the voice of a sharp quant analyst, keeping "
        "the same structure, section order and figures exactly as written — do not add figures, "
        "invent gaps, or change the noise/skill verdicts.\n\n" + assemble_quant_brief(facts)
    )


def generate_quant_brief(facts: PortfolioFacts) -> tuple[str, str]:
    """Produce ``(text, engine)`` for the quant-analyst brief, template-first.

    Deterministic template when no LLM key is configured; the provider-agnostic LLM rephrases it
    when a key is present, with the same output validation + redaction discipline as
    ``narrative.generate_brief``. Returns the redacted body and the engine tag — the dashboard
    renders it directly and nothing is persisted, so no schema is touched.
    """
    _validate_facts_keys(dict(facts))
    if llm.available():
        try:
            raw_text, engine = llm.complete(_build_prompt(facts), system=_SYSTEM, max_tokens=2048)
            rejection = _validate_llm_output(raw_text)
            if rejection is not None:
                log.warning(
                    "quant-analyst LLM brief rejected (%s); falling back to template",
                    rejection,
                )
                text = assemble_quant_brief(facts, note="LLM output failed validation")
                engine = "offline-template (llm-rejected)"
            else:
                text = raw_text
        except Exception as exc:  # noqa: BLE001 - GenAI is best-effort; template is the floor
            # redact: the provider key rides in the request URL/headers, so it can surface in the
            # httpx error string — never let it reach the logs (see utils/redact.py).
            log.warning(
                "quant-analyst LLM brief failed (%s); falling back to template",
                redact(str(exc)),
            )
            text = assemble_quant_brief(facts, note="LLM temporarily unavailable")
            engine = "offline-template (llm-failed)"
    else:
        text = assemble_quant_brief(facts)
        engine = "offline-template"
    log.info("quant-analyst brief generated via %s", engine)
    return redact(text), engine
