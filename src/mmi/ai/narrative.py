"""Generate a plain-English daily market brief from the marts, via the LLM layer.

Falls back to a deterministic template when no LLM key is configured, so the feature
always works (and CI/demo stay free).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mmi.ai import llm
from mmi.settings import settings
from mmi.utils.logging import get_logger

log = get_logger("ai.narrative")

_SYSTEM = (
    "You are a concise markets analyst. Given structured facts, write a 4-6 sentence daily "
    "brief for an informed reader. Be specific with numbers, neutral in tone, no financial "
    "advice, no hype. End with one sentence on what to watch."
)


def _q(con, sql: str) -> pd.DataFrame:
    try:
        return con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001 - missing table is fine pre-pipeline
        log.warning("fact query failed: %s", exc)
        return pd.DataFrame()


def gather_facts(con) -> dict:
    facts: dict = {"as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

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

    return facts


def _build_prompt(facts: dict) -> str:
    return "Here are today's structured market facts (JSON-like). Write the brief.\n\n" + str(facts)


def _offline_brief(facts: dict) -> str:
    """Deterministic template used when no LLM key is set."""
    lines = [
        f"**Market brief — {facts['as_of']}** _(template; set an LLM key for AI narrative)_",
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
    lines += ["", "_Watch: macro releases and any shift in the yield-curve spread._"]
    return "\n".join(lines)


def generate_brief(con) -> str:
    """Produce the brief, persist it to data/briefs/ and marts.market_brief."""
    facts = gather_facts(con)
    if llm.available():
        text = llm.complete(_build_prompt(facts), system=_SYSTEM)
        engine = llm.provider_model()
    else:
        text = _offline_brief(facts)
        engine = "offline-template"
    log.info("brief generated via %s", engine)

    # Persist to a dated markdown file (history of briefs).
    out_dir = Path(settings.duckdb_path).parent / "briefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    (out_dir / f"{stamp}.md").write_text(text, encoding="utf-8")

    # Persist to a mart for the dashboard.
    row = pd.DataFrame(
        [{"created_at": datetime.now(timezone.utc), "engine": engine, "brief": text}]
    )
    con.register("_brief", row)
    con.execute("CREATE TABLE IF NOT EXISTS marts.market_brief AS SELECT * FROM _brief LIMIT 0")
    con.execute("INSERT INTO marts.market_brief SELECT * FROM _brief")
    con.unregister("_brief")
    return text
