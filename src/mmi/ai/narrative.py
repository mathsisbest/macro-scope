"""Generate a plain-English daily market brief from the marts, via the LLM layer.

Falls back to a deterministic template when no LLM key is configured, so the feature
always works (and CI/demo stay free).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from mmi.ai import llm
from mmi.ml.skill_gate import skill_verdict
from mmi.portfolio import windows
from mmi.settings import settings
from mmi.utils.logging import get_logger
from mmi.utils.redact import redact

log = get_logger("ai.narrative")

# ---------------------------------------------------------------------------
# Facts TypedDict — contract-frozen key set.
#
# gather_facts() must return a dict whose keys are EXACTLY this set (some values
# are absent / None when the mart is missing, but no extra or missing keys).
# A contract test (see tests/test_ai_narrative.py) enforces this.
# ---------------------------------------------------------------------------


class FactsDict(TypedDict, total=False):
    """Typed contract for the facts dict produced by gather_facts().

    All keys are optional (total=False) because individual marts may be absent
    when running against an empty DB (CI, seed, demo).  The REQUIRED set is
    validated at runtime by _validate_facts_keys().
    """

    as_of: str
    data_date: str
    crypto: list[dict[str, Any]]
    spy: dict[str, Any]
    yields: dict[str, Any]
    portfolio: list[dict[str, Any]]
    portfolio_pairs: list[dict[str, Any]]
    ml_gate: dict[str, Any]
    vol_skill: dict[str, Any]


# The exact set of keys gather_facts() is allowed to produce.  An unexpected
# key is a contract violation; callers (template, LLM prompt) must not receive
# undocumented structure.
# Derived from FactsDict at import time so the contract is always in sync with the TypedDict;
# __annotations__ is the canonical runtime mapping for TypedDict (works on all CPython ≥3.10).
_FACTS_REQUIRED_KEYS: frozenset[str] = frozenset(FactsDict.__annotations__)


def _validate_facts_keys(facts: dict) -> None:
    """Raise ValueError if facts contains unknown keys or is missing required ones.

    'as_of' and 'data_date' are unconditionally produced; the rest are conditional
    on mart availability.  The FORBIDDEN direction is extra (undocumented) keys.
    """
    extra = set(facts.keys()) - _FACTS_REQUIRED_KEYS
    if extra:
        raise ValueError(
            f"gather_facts() produced unexpected keys not in the FactsDict contract: {extra!r}. "
            "Add them to FactsDict or remove them from gather_facts()."
        )


# ---------------------------------------------------------------------------
# LLM output validation — reject outputs that are empty, too long, or contain
# key-shaped tokens that a compromised/confused LLM might echo back.
# ---------------------------------------------------------------------------
_MAX_BRIEF_CHARS = 8000

# Patterns that signal a secret leaked into the LLM response body.
# Order matters: match the most-specific patterns first.
_SECRET_PATTERNS: list[re.Pattern] = [
    # URL query-param style: api_key=..., key=..., token=..., access_token=...
    re.compile(r"(?i)(?:api_?key|access_token|token|key)\s*=\s*\S+"),
    # Authorization: Bearer <token>
    re.compile(r"(?i)bearer\s+\S+"),
    # Google API key prefix (AIza...)
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    # Long hex strings (≥32 hex chars) that look like raw secret material.
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    # Long base64-ish tokens (≥32 chars, base64 alphabet including / and +).
    re.compile(r"[A-Za-z0-9+/]{32,}={0,2}"),
]


def _validate_llm_output(text: str) -> str | None:
    """Return a rejection reason string, or None if the output is acceptable.

    Rejects:
    * empty / whitespace-only responses
    * responses longer than _MAX_BRIEF_CHARS
    * responses containing key-shaped tokens (secret-leak guard)
    """
    if not text or not text.strip():
        return "LLM returned an empty/whitespace-only brief"

    if len(text) > _MAX_BRIEF_CHARS:
        return f"LLM brief is too long ({len(text)} chars > {_MAX_BRIEF_CHARS} limit)"

    for pat in _SECRET_PATTERNS:
        m = pat.search(text)
        if m:
            # Log the pattern name, never the matched value.
            return (
                f"LLM brief contains a key-shaped token matching pattern "
                f"{pat.pattern!r} — output rejected for safety"
            )

    return None


# The brief narrates the long-history baseline window (the dashboard default), so its portfolio
# facts are coherent now that the marts carry three windows. The BTC-effect (a cross-window
# comparison) is grounded separately.
_BRIEF_WINDOW = windows.DEFAULT_WINDOW

_SYSTEM = (
    "You are a concise markets analyst. Given pre-formatted market facts, write a 4-6 sentence "
    "daily brief for an informed reader. Use ONLY the figures provided, quoting them exactly as "
    "written — they are already formatted ($ prices, % returns/yields); do not add precision, "
    "restate raw decimals, or invent/estimate numbers. Be specific, neutral, no advice, no hype. "
    "If portfolio strategy stats are present, compare the allocation strategies against the 60/40 "
    "benchmark in one sentence; when a Sharpe confidence interval is wide or a pair is not "
    "distinguishable (see portfolio_pairs), say so plainly rather than implying an edge. "
    "If ml_gate is present, state in one sentence the mean weight the ML forecast earned in the "
    "mvo_ml blend and, when that weight is near zero, that the forecast showed no out-of-sample "
    "edge so mvo_ml tracked the historical-mean baseline. "
    "If vol_skill is present: when cleared=false, state plainly that the volatility model showed "
    "no demonstrated out-of-sample skill — do NOT use 'beats', 'outperforms', or similar phrasing; "
    "when cleared=true, you may state the model beat its persistence baseline out-of-sample. "
    "End with one sentence on what to watch."
)

_STRATEGY_LABELS = {
    "equal_weight": "Equal weight",
    "inverse_vol": "Inverse vol",
    "risk_parity": "Risk parity",
    "sixty_forty": "60/40 benchmark",
}


def _q(con, sql: str) -> pd.DataFrame:
    try:
        return con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001 - missing table is fine pre-pipeline
        log.warning("fact query failed: %s", exc)
        return pd.DataFrame()


def gather_facts(con) -> FactsDict:
    facts: dict = {"as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

    # Pull the most-recent data date from the asset mart so the brief header is grounded in
    # actual data, not wall-clock.  Falls back to as_of (wall-clock) when the mart is empty.
    data_date_row = _q(
        con,
        "select max(date) as data_date from marts.fct_asset_daily",
    )
    if not data_date_row.empty and data_date_row.iloc[0]["data_date"] is not None:
        raw = data_date_row.iloc[0]["data_date"]
        facts["data_date"] = str(raw)[:10]  # YYYY-MM-DD slice, handles both str and date/Timestamp
    else:
        facts["data_date"] = facts["as_of"][:10]

    # Crypto = BTC daily via Yahoo (marts.fct_asset_daily). The latest close + its daily return,
    # shaped with the same record keys (symbol/last_price/chg_24h) the template expects.
    crypto = _q(
        con,
        "select symbol, close as last_price, daily_return as chg_24h "
        "from marts.fct_asset_daily "
        "where asset_class = 'crypto' and symbol = 'BTC' order by date desc limit 1",
    )
    if not crypto.empty:
        facts["crypto"] = crypto.to_dict("records")

    spy = _q(
        con,
        "select date, close, daily_return, vol_20d from marts.fct_asset_daily "
        "where symbol = 'SPY' order by date desc limit 1",
    )
    if not spy.empty:
        facts["spy"] = spy.iloc[0].to_dict()

    curve = _q(
        con,
        "select us_10y, us_2y, yield_curve_10y_2y from marts.fct_market_macro "
        "order by date desc limit 1",
    )
    if not curve.empty:
        facts["yields"] = curve.iloc[0].to_dict()

    # Portfolio facts grounded in the real marts: returns-derived total return + worst drawdown,
    # joined to the bootstrap full-sample Sharpe + its confidence interval (one row per strategy).
    # Scoped to ONE window (the marts now carry three) so the per-strategy aggregates are coherent.
    portfolio = _q(
        con,
        f"""
        select s.strategy,
               agg.total_return,
               agg.max_drawdown,
               s.sharpe,
               s.sharpe_lo,
               s.sharpe_hi
        from marts.fct_portfolio_strategy_stats s
        join (
            select strategy,
                   arg_max(cumulative_return, date) as total_return,
                   min(drawdown) as max_drawdown
            from marts.fct_portfolio_returns
            where window_id = '{_BRIEF_WINDOW}'
            group by strategy
        ) agg on agg.strategy = s.strategy
        where s.window_id = '{_BRIEF_WINDOW}'
        order by s.sharpe desc
        """,
    )
    if not portfolio.empty:
        # Sort deterministically by strategy name so same facts → byte-identical offline brief
        # regardless of the order rows arrive from the mart.
        portfolio = portfolio.sort_values("strategy").reset_index(drop=True)
        facts["portfolio"] = portfolio.to_dict("records")

    # Pairwise distinguishability so the brief can hedge when differences are within noise.
    pairs = _q(
        con,
        "select strategy_a, strategy_b, distinguishable from marts.fct_portfolio_strategy_pairs "
        f"where window_id = '{_BRIEF_WINDOW}'",
    )
    if not pairs.empty:
        # Sort deterministically so same pairs in any row order → identical brief output.
        pairs = pairs.sort_values(["strategy_a", "strategy_b"]).reset_index(drop=True)
        facts["portfolio_pairs"] = pairs.to_dict("records")

    # The ML gate: the mean weight the forecast earned in mvo_ml's blend. A near-zero weight is the
    # honest signal that the ML showed no out-of-sample edge (so mvo_ml ≈ mvo_histmean).
    gate = _q(
        con,
        "select avg(forecast_weight) as mean_weight, max(forecast_weight) as max_weight "
        f"from marts.fct_portfolio_ml_gate where window_id = '{_BRIEF_WINDOW}'",
    )
    if not gate.empty and not pd.isna(gate.iloc[0]["mean_weight"]):
        facts["ml_gate"] = gate.iloc[0].to_dict()

    # Volatility model skill verdict — sources skill_verdict() which is the SINGLE source of the
    # gate verdict (Contract E). Reads model_metrics long-format rows; fails closed if missing.
    metrics_df = _q(con, "select model, symbol, metric, value from marts.model_metrics")
    verdict = skill_verdict(metrics_df)
    facts["vol_skill"] = {
        "cleared": verdict["cleared"],
        "oos_r2": verdict["oos_r2"],
        "reasons": verdict["reasons"],
    }

    # Contract enforcement: raise loudly if gather_facts() produced an undocumented key,
    # so regressions are caught at generation time rather than silently drifting.
    _validate_facts_keys(facts)
    return facts  # type: ignore[return-value]


def _n(x: object, default: float = 0.0) -> float:
    """Coerce a possibly-None/NaN numeric fact to a plain float (NaN is truthy, so guard it)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    return float(x)  # type: ignore[arg-type]


def _fmt_facts_for_prompt(facts: FactsDict) -> str:
    """Render the facts as a clean, pre-formatted block for the LLM — $ prices, % returns/yields,
    2dp Sharpe — so the model quotes tidy figures instead of echoing raw floats (e.g.
    ``59362.21875`` / ``-0.006018...``). The offline template formats independently from the raw
    facts dict; this is the LLM-prompt view only."""
    lines: list[str] = [f"Data as of {facts.get('data_date', facts.get('as_of', ''))}."]
    for c in facts.get("crypto", []):
        lines.append(
            f"{str(c['symbol']).title()}: ${_n(c.get('last_price')):,.0f} "
            f"({_n(c.get('chg_24h')):+.2%} on the day)."
        )
    if "spy" in facts:
        s = facts["spy"]
        line = (
            f"SPY: close ${_n(s.get('close')):,.2f} ({_n(s.get('daily_return')):+.2%} on the day)"
        )
        if s.get("vol_20d") is not None and not pd.isna(s["vol_20d"]):
            line += f", 20-day annualised vol {_n(s.get('vol_20d')):.1%}"
        lines.append(line + ".")
    if "yields" in facts:
        y = facts["yields"]
        parts = []
        if y.get("us_10y") is not None:
            parts.append(f"10Y {_n(y.get('us_10y')):.2f}%")
        if y.get("us_2y") is not None:
            parts.append(f"2Y {_n(y.get('us_2y')):.2f}%")
        sp = y.get("yield_curve_10y_2y")
        tail = f" -> 10Y-2Y spread {_n(sp):+.2f}pp" if sp is not None else ""
        if parts:
            lines.append("Treasuries: " + " / ".join(parts) + tail + ".")
    if facts.get("portfolio"):
        lines.append("Portfolio strategies (long-baseline window):")
        for p in facts["portfolio"]:
            lines.append(
                f"  - {_STRATEGY_LABELS.get(p['strategy'], p['strategy'])}: "
                f"total return {_n(p.get('total_return')):+.1%}, "
                f"max drawdown {_n(p.get('max_drawdown')):.1%}, {_sharpe_phrase(p)}."
            )
    if facts.get("portfolio_pairs"):
        pairs = facts["portfolio_pairs"]
        n_distinct = sum(1 for p in pairs if p.get("distinguishable"))
        lines.append(
            f"Pairwise Sharpe distinguishability: {n_distinct} of {len(pairs)} pairs "
            "distinguishable at the 90% level."
        )
    if "ml_gate" in facts:
        mw = facts["ml_gate"].get("mean_weight")
        if mw is not None and not pd.isna(mw):
            lines.append(
                f"ML forecast mean weight in mvo_ml: {_n(mw):.1%} (near zero = no OOS edge)."
            )
    if "vol_skill" in facts:
        v = facts["vol_skill"]
        r2 = v.get("oos_r2")
        r2s = f"{r2:.3f}" if isinstance(r2, (int, float)) and not pd.isna(r2) else "n/a"
        lines.append(f"Volatility model: cleared={v.get('cleared')}, out-of-sample R2={r2s}.")
    return "\n".join(lines)


def _build_prompt(facts: FactsDict) -> str:
    return (
        "Here are today's market facts, already formatted for you. Write the brief, quoting these "
        "figures exactly as written.\n\n" + _fmt_facts_for_prompt(facts)
    )


def _sharpe_phrase(p: dict) -> str:
    """'Sharpe X [lo, hi]' from the bootstrap stats, degrading gracefully if a value is absent."""
    sharpe = p.get("sharpe")
    if sharpe is None or pd.isna(sharpe):
        return "Sharpe n/a"
    lo, hi = p.get("sharpe_lo"), p.get("sharpe_hi")
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return f"Sharpe {sharpe:.2f}"
    return f"Sharpe {sharpe:.2f} [{lo:.2f}, {hi:.2f}]"


def _distinguishability_note(pairs: list) -> str:
    """One honest sentence: are any strategy Sharpe differences beyond bootstrap noise?"""
    if not pairs:
        return ""
    distinct = [p for p in pairs if p["distinguishable"]]
    if not distinct:
        return (
            "_Statistical note: at the 90% level no pair of strategies is distinguishable by "
            "Sharpe — the differences are within bootstrap noise._"
        )
    # Sort deterministically so same pairs in any input order produce identical output.
    sorted_distinct = sorted(distinct, key=lambda p: (p["strategy_a"], p["strategy_b"]))
    named = ", ".join(
        f"{_STRATEGY_LABELS.get(p['strategy_a'], p['strategy_a'])} vs "
        f"{_STRATEGY_LABELS.get(p['strategy_b'], p['strategy_b'])}"
        for p in sorted_distinct
    )
    return f"_Statistical note: only {named} differ beyond bootstrap noise (90% CI)._"


def _offline_brief(facts: FactsDict, note: str = "no LLM key set") -> str:
    """Deterministic template used when the LLM narrative is unavailable. ``note`` states *why*
    (no key, the call failed, or the output was rejected) so the header stays honest — the old
    text always said "set an LLM key", which was misleading when a key WAS set but the call 503'd.
    """
    # Use the data date (YYYY-MM-DD from the marts) in the header so it is grounded in the
    # actual data window, not the wall-clock generation time (Contract G).
    data_date = facts.get("data_date", facts.get("as_of", ""))
    lines = [
        f"**Market brief — data as of {data_date}** _(deterministic template — {note})_",
        "",
    ]
    for c in facts.get("crypto", []):
        chg = (c.get("chg_24h") or 0) * 100
        lines.append(f"- {c['symbol'].title()}: ${c['last_price']:,.0f} ({chg:+.1f}% 1d)")
    if "spy" in facts:
        s = facts["spy"]
        ret = (s.get("daily_return") or 0) * 100
        lines.append(f"- SPY last close ${s['close']:,.2f} ({ret:+.2f}% on the day).")
    if "yields" in facts and facts["yields"].get("yield_curve_10y_2y") is not None:
        y = facts["yields"]
        spread = y["yield_curve_10y_2y"]
        lines.append(
            f"- 10Y {y['us_10y']:.2f}% / 2Y {y['us_2y']:.2f}% → 10Y-2Y spread {spread:+.2f}pp."
        )
    if facts.get("portfolio"):
        lines += ["", "**Strategy comparison** (walk-forward, net of costs):"]
        # Sort deterministically by strategy name so the same facts produce a byte-identical
        # brief regardless of the order the caller assembled the list.
        for p in sorted(facts["portfolio"], key=lambda x: x.get("strategy", "")):
            name = _STRATEGY_LABELS.get(p["strategy"], p["strategy"])
            lines.append(
                f"- {name}: {p['total_return'] * 100:+.1f}% total return, "
                f"max drawdown {p['max_drawdown'] * 100:.1f}%, {_sharpe_phrase(p)}."
            )
        note = _distinguishability_note(facts.get("portfolio_pairs", []))
        if note:
            lines.append(note)
        gate = facts.get("ml_gate")
        if gate and gate.get("mean_weight") is not None:
            weight = gate["mean_weight"]
            if weight < 0.05:
                lines.append(
                    f"- ML gate: the forecast earned a mean weight of {weight:.0%} in mvo_ml's "
                    "blend — no reliable out-of-sample edge, so mvo_ml tracked the historical-mean "
                    "baseline."
                )
            else:
                lines.append(
                    f"- ML gate: the forecast earned a mean weight of {weight:.0%} in mvo_ml's "
                    "blend over the historical-mean prior."
                )
    # Volatility model skill verdict — honest escape-hatch state (Contract E / Contract G).
    vol_skill = facts.get("vol_skill")
    if vol_skill is not None:
        if vol_skill.get("cleared"):
            r2 = vol_skill.get("oos_r2")
            r2_str = f" (OOS R² {r2:.2f})" if r2 is not None else ""
            lines.append(
                f"- Volatility model (HAR/SPY): the model beat its persistence baseline "
                f"out-of-sample{r2_str}."
            )
        else:
            lines.append(
                "- Volatility model (HAR/SPY): the model showed no demonstrated "
                "out-of-sample skill — no reliable out-of-sample edge; baseline-only state."
            )
    lines += ["", "_Watch: macro releases and any shift in the yield-curve spread._"]
    return "\n".join(lines)


def generate_brief(con) -> str:
    """Produce the brief, persist it to data/briefs/ and marts.market_brief."""
    facts = gather_facts(con)
    if llm.available():
        try:
            # 2048 (not the 800 default) so medium thinking has room before the answer.
            raw_text = llm.complete(_build_prompt(facts), system=_SYSTEM, max_tokens=4096)
            rejection = _validate_llm_output(raw_text)
            if rejection is not None:
                # Log the rejection reason (no secret values in there — they matched by
                # pattern, not value) and fall back to the offline template.
                log.warning(
                    "LLM brief rejected (%s); falling back to offline template",
                    rejection,
                )
                text = _offline_brief(facts, note="LLM output failed validation")
                engine = "offline-template (llm-rejected)"
            else:
                text = raw_text
                engine = llm.provider_model()
        except Exception as exc:  # noqa: BLE001 - GenAI is best-effort; template is the floor
            # redact: the provider key rides in the request URL/headers, so it can surface in the
            # httpx error string — never let it reach the logs (see utils/redact.py).
            log.warning("LLM brief failed (%s); falling back to offline template", redact(str(exc)))
            text = _offline_brief(facts, note="LLM temporarily unavailable")
            engine = "offline-template (llm-failed)"
    else:
        text = _offline_brief(facts)
        engine = "offline-template"
    log.info("brief generated via %s", engine)

    # Contract G: redact() EVERY persisted brief body before it is written to the .md file
    # or the mart — closes the P1 where a raw (potentially key-bearing) body was stored.
    safe_text = redact(text)

    # Persist to a dated markdown file (history of briefs).
    out_dir = Path(settings.duckdb_path).parent / "briefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    (out_dir / f"{stamp}.md").write_text(safe_text, encoding="utf-8")

    # Persist to a mart for the dashboard.
    row = pd.DataFrame(
        [{"created_at": datetime.now(timezone.utc), "engine": engine, "brief": safe_text}]
    )
    con.register("_brief", row)
    con.execute("CREATE TABLE IF NOT EXISTS marts.market_brief AS SELECT * FROM _brief LIMIT 0")
    con.execute("INSERT INTO marts.market_brief SELECT * FROM _brief")
    con.unregister("_brief")
    return safe_text
