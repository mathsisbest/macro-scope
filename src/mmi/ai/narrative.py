"""Generate a plain-English daily market brief from the marts, via the LLM layer.

Falls back to a deterministic template when no LLM key is configured, so the feature
always works (and CI/demo stay free).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mmi.ai import llm
from mmi.ml.skill_gate import skill_verdict
from mmi.portfolio import windows
from mmi.settings import settings
from mmi.utils.logging import get_logger
from mmi.utils.redact import redact

log = get_logger("ai.narrative")

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
    "You are a concise markets analyst. Given structured facts (JSON-like), write a 4-6 sentence "
    "daily brief for an informed reader. Use ONLY the numbers present in the facts — never invent, "
    "estimate, or round away figures. Be specific, neutral in tone, no financial advice, no hype. "
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


def gather_facts(con) -> dict:
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

    crypto = _q(
        con,
        """
        with ranked as (
            select symbol, ts, price_usd,
                   row_number() over (partition by symbol order by ts desc) rn
            from marts.fct_crypto_intraday
        )
        select a.symbol, a.price_usd as last_price,
               a.price_usd / b.price_usd - 1 as chg_24h
        from ranked a join ranked b on a.symbol = b.symbol and b.rn = 25
        where a.rn = 1
        """,
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
        facts["portfolio"] = portfolio.to_dict("records")

    # Pairwise distinguishability so the brief can hedge when differences are within noise.
    pairs = _q(
        con,
        "select strategy_a, strategy_b, distinguishable from marts.fct_portfolio_strategy_pairs "
        f"where window_id = '{_BRIEF_WINDOW}'",
    )
    if not pairs.empty:
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

    return facts


def _build_prompt(facts: dict) -> str:
    return "Here are today's structured market facts (JSON-like). Write the brief.\n\n" + str(facts)


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
    named = ", ".join(
        f"{_STRATEGY_LABELS.get(p['strategy_a'], p['strategy_a'])} vs "
        f"{_STRATEGY_LABELS.get(p['strategy_b'], p['strategy_b'])}"
        for p in distinct
    )
    return f"_Statistical note: only {named} differ beyond bootstrap noise (90% CI)._"


def _offline_brief(facts: dict) -> str:
    """Deterministic template used when no LLM key is set."""
    # Use the data date (YYYY-MM-DD from the marts) in the header so it is grounded in the
    # actual data window, not the wall-clock generation time (Contract G).
    data_date = facts.get("data_date", facts.get("as_of", ""))
    lines = [
        f"**Market brief — data as of {data_date}** _(template; set an LLM key for AI narrative)_",
        "",
    ]
    for c in facts.get("crypto", []):
        lines.append(
            f"- {c['symbol'].title()}: ${c['last_price']:,.0f} ({c['chg_24h'] * 100:+.1f}% 24h)"
        )
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
        for p in facts["portfolio"]:
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
            raw_text = llm.complete(_build_prompt(facts), system=_SYSTEM, max_tokens=2048)
            rejection = _validate_llm_output(raw_text)
            if rejection is not None:
                # Log the rejection reason (no secret values in there — they matched by
                # pattern, not value) and fall back to the offline template.
                log.warning(
                    "LLM brief rejected (%s); falling back to offline template",
                    rejection,
                )
                text = _offline_brief(facts)
                engine = "offline-template (llm-rejected)"
            else:
                text = raw_text
                engine = llm.provider_model()
        except Exception as exc:  # noqa: BLE001 - GenAI is best-effort; template is the floor
            # redact: the provider key rides in the request URL/headers, so it can surface in the
            # httpx error string — never let it reach the logs (see utils/redact.py).
            log.warning("LLM brief failed (%s); falling back to offline template", redact(str(exc)))
            text = _offline_brief(facts)
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
