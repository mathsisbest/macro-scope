"""Generate a plain-English daily macro & markets narrative from the marts, via the LLM layer.

The brief is market commentary: it tells the story of the day's macro backdrop and cross-asset
moves — what rose and fell together, what diverged, where there's momentum / mean-reversion /
rotation, and the plausible *why* tied to rates, inflation, the dollar, oil and risk appetite.
It deliberately does NOT discuss portfolio strategies, allocation, or the ML/volatility model —
those live on their own dashboard tabs.

Falls back to a deterministic template when no LLM key is configured (or the call fails / its
output is rejected), so the feature always works and CI/demo stay free.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from mmi.ai import llm
from mmi.settings import settings
from mmi.utils.logging import get_logger
from mmi.utils.redact import redact

log = get_logger("ai.narrative")

# ---------------------------------------------------------------------------
# Facts TypedDict — contract-frozen key set.
#
# gather_facts() must return a dict whose keys are EXACTLY this set (some values
# are absent when the mart is missing, but no extra or missing keys). A contract
# test (see tests/test_ai_narrative.py) enforces this.
# ---------------------------------------------------------------------------


class FactsDict(TypedDict, total=False):
    """Typed contract for the facts dict produced by gather_facts().

    All keys are optional (total=False) because individual marts may be absent
    when running against an empty DB (CI, seed, demo). 'as_of'/'data_date' are
    always produced; the rest are conditional on mart availability.
    """

    as_of: str
    data_date: str
    macro: list[dict[str, Any]]
    assets: list[dict[str, Any]]
    correlations: list[dict[str, Any]]


_FACTS_REQUIRED_KEYS: frozenset[str] = frozenset(FactsDict.__annotations__)


def _validate_facts_keys(facts: dict) -> None:
    """Raise ValueError if facts contains unknown keys (extra/undocumented structure)."""
    extra = set(facts.keys()) - _FACTS_REQUIRED_KEYS
    if extra:
        raise ValueError(
            f"gather_facts() produced unexpected keys not in the FactsDict contract: {extra!r}. "
            "Add them to FactsDict or remove them from gather_facts()."
        )


# ---------------------------------------------------------------------------
# Curated panels — the editorial selection of what the brief narrates.
# ---------------------------------------------------------------------------

# Macro series (FRED ids) the brief reports, in narrative order, with display label + units.
# A deliberate, readable subset of the full config/assets.yml macro universe.
_MACRO_BRIEF: list[tuple[str, str, str]] = [
    ("VIXCLS", "VIX (equity volatility)", "index"),
    ("DGS3MO", "3M Treasury yield", "%"),
    ("DGS2", "2Y Treasury yield", "%"),
    ("DGS10", "10Y Treasury yield", "%"),
    ("T10Y2Y", "10Y-2Y curve", "pp"),
    ("FEDFUNDS", "Fed funds rate", "%"),
    ("CPIAUCSL", "CPI inflation", "yoy"),
    ("PCEPILFE", "Core PCE inflation", "yoy"),
    ("UNRATE", "Unemployment rate", "%"),
    ("A191RL1Q225SBEA", "Real GDP growth (ann.)", "%"),
    ("DCOILWTICO", "WTI crude oil", "$/bbl"),
    ("DTWEXBGS", "US dollar (broad index)", "index"),
    ("NFCI", "Financial conditions (NFCI)", "index"),
]

# Series rendered as a 12-month % change (year-over-year) rather than a raw index level — for
# these an index value (e.g. CPI = 334) is meaningless to a reader; the YoY rate is the signal.
_YOY_SERIES: frozenset[str] = frozenset({"CPIAUCSL", "PCEPILFE"})

# Friendly labels for the tracked assets (asset_class comes from the mart).
_ASSET_LABELS: dict[str, str] = {
    "SPY": "S&P 500 (SPY)",
    "QQQ": "Nasdaq 100 (QQQ)",
    "VEA": "Developed ex-US equities (VEA)",
    "TLT": "Long Treasuries (TLT)",
    "TIP": "TIPS (TIP)",
    "GLD": "Gold (GLD)",
    "BTC": "Bitcoin",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
}

# Rolling window (trading days) for the "recent co-movement" correlation read.
_CORR_WINDOW: int = 60
_CORR_MIN_OBS: int = 30

# ---------------------------------------------------------------------------
# LLM output validation — reject empty / too-long / key-shaped responses.
# ---------------------------------------------------------------------------
_MAX_BRIEF_CHARS = 8000

_SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(?:api_?key|access_token|token|key)\s*=\s*\S+"),
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    re.compile(r"[A-Za-z0-9+/]{32,}={0,2}"),
]


def _validate_llm_output(text: str) -> str | None:
    """Return a rejection reason string, or None if the output is acceptable."""
    if not text or not text.strip():
        return "LLM returned an empty/whitespace-only brief"
    if len(text) > _MAX_BRIEF_CHARS:
        return f"LLM brief is too long ({len(text)} chars > {_MAX_BRIEF_CHARS} limit)"
    for pat in _SECRET_PATTERNS:
        m = pat.search(text)
        if m:
            return (
                f"LLM brief contains a key-shaped token matching pattern "
                f"{pat.pattern!r} — output rejected for safety"
            )
    return None


_SYSTEM = (
    "You are a sharp, experienced markets commentator writing a daily macro-and-markets "
    "narrative for a sophisticated reader who watches Bloomberg, reads the FT, and thinks in "
    "terms of regimes, carry, and positioning.\n\n"
    "Write 12-18 sentences. Open with the single most important thing that happened today — "
    "the headline. Then unpack it: what moved, by how much, and the plausible mechanism tying "
    "it to the macro backdrop (rates, the yield curve, inflation expectations, the Fed, the "
    "dollar, oil, and risk appetite via the VIX).\n\n"
    "Use cross-asset moves to tell a story about regime and positioning:\n"
    "- When 1d/5d/20d returns point the same way, call it momentum.\n"
    "- When a move snaps back toward the 50-day average, call it mean-reversion.\n"
    "- When money shifts between asset classes or regions (tech vs broad equities, stocks vs "
    "bonds vs gold vs bitcoin), describe it as rotation.\n"
    "- Reference the correlation data to support claims about co-movement or divergence.\n\n"
    "Layer in context: where are we in the rate cycle? Is the yield curve steepening or "
    "flattening? What does the VIX regime imply about positioning? Is the dollar acting as "
    "a safe haven or a risk-on signal? What's the oil story telling us about global demand?\n\n"
    "Frame causation cautiously ('consistent with', 'amid', 'in a pattern consistent with') "
    "— never state causation as certainty. Use ONLY the figures provided and quote them as "
    "written (already formatted as %, $, pp). Do not invent, estimate, or add precision.\n\n"
    "Do NOT mention portfolio strategies, asset allocation, Sharpe ratios, backtests, or any "
    "machine-learning or volatility-model forecast — this is market commentary only.\n\n"
    "Tone: direct, opinionated but grounded, no hype, no filler. Think Matt Levine meets "
    "a central bank research note. End with 1-2 sentences on what to watch next and why it "
    "matters."
)


def _q(con, sql: str) -> pd.DataFrame:
    try:
        return con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001 - missing table is fine pre-pipeline
        log.warning("fact query failed: %s", exc)
        return pd.DataFrame()


def _n(x: object, default: float = 0.0) -> float:
    """Coerce a possibly-None/NaN numeric fact to a plain float (NaN is truthy, so guard it)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    return float(x)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fact gathering
# ---------------------------------------------------------------------------


def _yoy_change(con, series_id: str) -> float | None:
    """Year-over-year fractional change for a (monthly) macro series, or None if unavailable.

    Compares the latest value to the last value at or before ~12 months earlier. Returns None
    when there isn't ~a year of history (e.g. small sample data) or the prior value is 0/NaN.
    """
    df = _q(
        con,
        f"select date, value from marts.fct_macro_indicator where series_id = '{series_id}' "
        "order by date",
    )
    if df.empty or len(df) < 13:
        return None
    latest = df.iloc[-1]
    cutoff = pd.Timestamp(latest["date"]) - pd.DateOffset(months=12)
    prior = df[pd.to_datetime(df["date"]) <= cutoff]
    if prior.empty:
        return None
    pv = prior["value"].iloc[-1]
    if pv is None or pd.isna(pv) or pv == 0 or pd.isna(latest["value"]):
        return None
    return float(latest["value"] / pv - 1.0)


def _macro_readings(con) -> list[dict[str, Any]]:
    """Curated macro panel: latest value + change per series, plus YoY for inflation series."""
    latest = _q(
        con,
        "select series_id, value, change from marts.fct_macro_indicator "
        "qualify row_number() over (partition by series_id order by date desc) = 1",
    )
    if latest.empty:
        return []
    by_id = {str(r["series_id"]): r for _, r in latest.iterrows()}
    out: list[dict[str, Any]] = []
    for sid, label, units in _MACRO_BRIEF:
        row = by_id.get(sid)
        if row is None:
            continue
        rec: dict[str, Any] = {
            "series_id": sid,
            "label": label,
            "units": units,
            "value": _n(row["value"], default=float("nan")),
            "change": None if pd.isna(row["change"]) else float(row["change"]),
        }
        if units == "yoy":
            rec["yoy"] = _yoy_change(con, sid)
        out.append(rec)
    return out


def _asset_signals(con) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Per-asset momentum/mean-reversion signals + a wide daily-return frame for correlation.

    Returns (assets, returns_wide). ``assets`` carries 1d/5d/20d returns, position vs the 50-day
    MA, and 20d vol per symbol. ``returns_wide`` is the last ``_CORR_WINDOW`` days of daily
    returns pivoted date×symbol, for :func:`_correlation_notes`.
    """
    df = _q(
        con,
        "select symbol, asset_class, date, close, daily_return, ma_50, vol_20d "
        "from marts.fct_asset_daily "
        "qualify row_number() over (partition by symbol order by date desc) <= 70 "
        "order by symbol, date",
    )
    if df.empty:
        return [], pd.DataFrame()

    assets: list[dict[str, Any]] = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        close = g["close"].to_numpy(dtype=float)
        if len(close) < 2:
            continue
        ma50 = g["ma_50"].iloc[-1] if "ma_50" in g.columns else None
        last_dr = g["daily_return"].iloc[-1]
        assets.append(
            {
                "symbol": str(sym),
                "label": _ASSET_LABELS.get(str(sym), str(sym)),
                "asset_class": str(g["asset_class"].iloc[0]),
                "last_close": float(close[-1]),
                "ret_1d": None if pd.isna(last_dr) else float(last_dr),
                "ret_5d": float(close[-1] / close[-6] - 1.0) if len(close) >= 6 else None,
                "ret_20d": float(close[-1] / close[-21] - 1.0) if len(close) >= 21 else None,
                "vs_ma50": (
                    float(close[-1] / float(ma50) - 1.0)
                    if ma50 is not None and not pd.isna(ma50) and float(ma50) != 0.0
                    else None
                ),
                "vol_20d": (
                    None if pd.isna(g["vol_20d"].iloc[-1]) else float(g["vol_20d"].iloc[-1])
                ),
            }
        )

    wide = (
        df.pivot_table(index="date", columns="symbol", values="daily_return")
        .sort_index()
        .tail(_CORR_WINDOW)
    )
    return assets, wide


def _correlation_notes(returns_wide: pd.DataFrame) -> list[dict[str, Any]]:
    """Notable recent daily-return correlations: stocks-vs-bonds, most-positive, most-negative."""
    if returns_wide.empty or returns_wide.shape[0] < _CORR_MIN_OBS or returns_wide.shape[1] < 2:
        return []
    corr = returns_wide.corr(min_periods=_CORR_MIN_OBS // 2)
    notes: list[dict[str, Any]] = []

    if {"SPY", "TLT"} <= set(corr.columns) and not pd.isna(corr.loc["SPY", "TLT"]):
        notes.append(
            {
                "a": "SPY",
                "b": "TLT",
                "corr": float(corr.loc["SPY", "TLT"]),
                "kind": "stocks vs bonds",
            }
        )

    cols = list(corr.columns)
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if not pd.isna(v):
                pairs.append((str(cols[i]), str(cols[j]), float(v)))
    if pairs:
        pairs.sort(key=lambda t: t[2])
        lo, hi = pairs[0], pairs[-1]
        # Sign-neutral labels: the extremes may both be positive in a risk-on regime, so "highest"
        # / "lowest" stays accurate (the signed value is right there for the reader/LLM to judge).
        notes.append({"a": hi[0], "b": hi[1], "corr": hi[2], "kind": "highest co-movement"})
        notes.append({"a": lo[0], "b": lo[1], "corr": lo[2], "kind": "lowest co-movement"})
    return notes


def gather_facts(con) -> FactsDict:
    facts: dict = {"as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

    data_date_row = _q(con, "select max(date) as data_date from marts.fct_asset_daily")
    if not data_date_row.empty and data_date_row.iloc[0]["data_date"] is not None:
        facts["data_date"] = str(data_date_row.iloc[0]["data_date"])[:10]
    else:
        facts["data_date"] = facts["as_of"][:10]

    macro = _macro_readings(con)
    if macro:
        facts["macro"] = macro

    assets, returns_wide = _asset_signals(con)
    if assets:
        facts["assets"] = assets
    corr = _correlation_notes(returns_wide)
    if corr:
        facts["correlations"] = corr

    _validate_facts_keys(facts)
    return facts  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Formatting — shared by the LLM prompt and the deterministic offline template.
# ---------------------------------------------------------------------------


def _fmt_macro_value(rec: dict[str, Any]) -> str | None:
    """Format one macro reading's headline value by units. None if it can't be rendered."""
    units = rec.get("units")
    if units == "yoy":
        yoy = rec.get("yoy")
        if yoy is None or pd.isna(yoy):
            return None  # not enough history for a YoY read → skip the line entirely
        return f"{yoy:+.1%} YoY"
    value = rec.get("value")
    if value is None or pd.isna(value):
        return None
    v = float(value)
    if units == "%":
        return f"{v:.2f}%"
    if units == "pp":
        return f"{v:+.2f}pp"
    if units == "$/bbl":
        return f"${v:,.2f}"
    if units == "index":
        return f"{v:,.1f}"
    return f"{v:,.2f}"


def _fmt_macro_change(rec: dict[str, Any]) -> str:
    """Short change suffix for a macro reading (empty when no change / YoY series)."""
    if rec.get("units") == "yoy":
        return ""
    chg = rec.get("change")
    if chg is None or pd.isna(chg):
        return ""
    return f" (Δ {float(chg):+,.2f})"


def _macro_lines(macro: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for rec in macro:
        disp = _fmt_macro_value(rec)
        if disp is None:
            continue
        lines.append(f"- {rec['label']}: {disp}{_fmt_macro_change(rec)}")
    return lines


def _asset_line(a: dict[str, Any]) -> str:
    parts: list[str] = []
    if a.get("ret_1d") is not None:
        parts.append(f"1d {_n(a['ret_1d']):+.1%}")
    if a.get("ret_5d") is not None:
        parts.append(f"5d {_n(a['ret_5d']):+.1%}")
    if a.get("ret_20d") is not None:
        parts.append(f"20d {_n(a['ret_20d']):+.1%}")
    if a.get("vs_ma50") is not None:
        parts.append(f"{_n(a['vs_ma50']):+.1%} vs 50d avg")
    if a.get("vol_20d") is not None:
        parts.append(f"20d vol {_n(a['vol_20d']):.1%}")
    return f"- {a['label']} [{a['asset_class']}]: " + ", ".join(parts)


def _corr_lines(correlations: list[dict[str, Any]]) -> list[str]:
    return [f"- {c['a']}–{c['b']} {_n(c['corr']):+.2f} ({c['kind']})" for c in correlations]


def _fmt_facts_for_prompt(facts: FactsDict) -> str:
    """Render the facts as a clean, pre-formatted block (%, $, pp) so the LLM quotes tidy figures
    instead of echoing raw floats. The offline template formats independently of this."""
    out: list[str] = [f"Data as of {facts.get('data_date', facts.get('as_of', ''))}."]

    macro_lines = _macro_lines(facts.get("macro", []))
    if macro_lines:
        out += ["", "Macro backdrop (latest readings):", *macro_lines]

    assets = facts.get("assets", [])
    if assets:
        out += ["", "Cross-asset moves:", *[_asset_line(a) for a in assets]]

    corr_lines = _corr_lines(facts.get("correlations", []))
    if corr_lines:
        out += [
            "",
            f"Recent co-movement (~{_CORR_WINDOW}-day daily-return correlation):",
            *corr_lines,
        ]
    return "\n".join(out)


def _build_prompt(facts: FactsDict) -> str:
    return (
        "Here are today's macro and market readings, already formatted for you. Write the daily "
        "narrative described in your instructions, quoting these figures exactly as written.\n\n"
        + _fmt_facts_for_prompt(facts)
    )


def _offline_brief(facts: FactsDict, note: str = "no LLM key set") -> str:
    """Deterministic template used when the LLM narrative is unavailable.

    It reports the macro backdrop, cross-asset moves and notable correlations honestly and
    deterministically (same facts → byte-identical output) — but does not editorialise the
    'why' the way the LLM narrative does.
    """
    data_date = facts.get("data_date", facts.get("as_of", ""))
    lines = [
        f"**Market brief — data as of {data_date}** _(deterministic template — {note})_",
        "",
    ]

    macro_lines = _macro_lines(facts.get("macro", []))
    if macro_lines:
        lines += ["**Macro backdrop**", *macro_lines, ""]

    assets = facts.get("assets", [])
    if assets:
        lines += ["**Cross-asset moves**", *[_asset_line(a) for a in assets], ""]

    corr_lines = _corr_lines(facts.get("correlations", []))
    if corr_lines:
        lines += [
            f"**Recent co-movement** (~{_CORR_WINDOW}-day daily-return correlation)",
            *corr_lines,
            "",
        ]

    lines.append("_Watch: incoming macro releases and any shift in rates, the dollar, or risk._")
    return "\n".join(lines)


def generate_brief(con) -> str:
    """Produce the brief, persist it to data/briefs/ and marts.market_brief."""
    facts = gather_facts(con)
    if llm.available():
        try:
            # 4096 so medium thinking has room before the 6-10 sentence narrative answer.
            raw_text = llm.complete(_build_prompt(facts), system=_SYSTEM, max_tokens=4096)
            rejection = _validate_llm_output(raw_text)
            if rejection is not None:
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

    # Contract G: redact() EVERY persisted brief body before it is written to the .md file or mart.
    safe_text = redact(text)

    out_dir = Path(settings.duckdb_path).parent / "briefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    (out_dir / f"{stamp}.md").write_text(safe_text, encoding="utf-8")

    try:
        data_date = (
            con.execute("select cast(max(date) as varchar) from marts.fct_asset_daily").fetchone()[
                0
            ]
            or ""
        )
    except Exception:
        data_date = ""
    row = pd.DataFrame(
        [
            {
                "created_at": datetime.now(timezone.utc),
                "engine": engine,
                "brief": safe_text,
                "data_date": data_date,
            }
        ]
    )
    con.register("_brief", row)
    con.execute("CREATE TABLE IF NOT EXISTS marts.market_brief AS SELECT * FROM _brief LIMIT 0")
    con.execute("INSERT INTO marts.market_brief SELECT * FROM _brief")
    con.unregister("_brief")
    return safe_text
