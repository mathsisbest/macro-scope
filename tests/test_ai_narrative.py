"""The market brief is a numerically-grounded macro & markets NARRATIVE.

Covers:
- gather_facts(): curated macro panel (+ YoY inflation), per-asset momentum/MA signals, recent
  cross-asset correlations — and NO portfolio / ML / vol-model facts (those moved off the brief).
- the LLM prompt presents pre-formatted figures (%, $, pp) and forbids strategy/ML content.
- post-generation LLM output validation (empty / too-long / key-shaped → llm-rejected tag).
- body redaction: redact() runs on EVERY persisted body before .md write + mart insert.
- HTTP 429 / API error → 'offline-template (llm-failed)', not a crash; engine tags distinct.
- gather_facts() TypedDict key-set contract — an unexpected key raises ValueError.
"""

import logging
import re

import duckdb
import httpx
import numpy as np
import pandas as pd

from mmi.ai import narrative

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_minimal_con() -> duckdb.DuckDBPyConnection:
    """Marts schema only (no tables) — gather_facts must degrade, not crash."""
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    return con


def _rich_con() -> duckdb.DuckDBPyConnection:
    """In-memory DB with realistic fct_asset_daily + fct_macro_indicator marts.

    60 business days × 4 assets (enough for 20d returns + the 60d correlation window), and 15
    monthly rows per macro series (enough for a YoY inflation read).
    """
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    rng = np.random.default_rng(7)

    # --- assets ---
    dates = pd.bdate_range("2024-01-01", periods=60)
    classes = {"SPY": "equities", "QQQ": "equities", "TLT": "bonds", "GLD": "commodities"}
    frames = []
    for sym, cls in classes.items():
        rets = rng.normal(0.0004, 0.01, len(dates))
        close = 100.0 * np.cumprod(1 + rets)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "asset_class": cls,
                    "date": dates,
                    "close": close,
                    "daily_return": np.concatenate([[0.0], np.diff(close) / close[:-1]]),
                    "ma_50": close * 0.98,  # price ~+2% vs its 50d average
                    "vol_20d": np.abs(rng.normal(0.01, 0.002, len(dates))),
                }
            )
        )
    assets = pd.concat(frames, ignore_index=True)
    con.register("_a", assets)
    con.execute("create table marts.fct_asset_daily as select * from _a")
    con.unregister("_a")

    # --- macro (15 monthly rows per curated series so latest + YoY both resolve) ---
    months = pd.date_range("2024-06-01", periods=15, freq="MS")
    base = {
        "VIXCLS": 18.0,
        "DGS3MO": 3.8,
        "DGS2": 4.1,
        "DGS10": 4.4,
        "T10Y2Y": 0.3,
        "FEDFUNDS": 3.6,
        "CPIAUCSL": 300.0,
        "PCEPILFE": 120.0,
        "UNRATE": 4.3,
        "A191RL1Q225SBEA": 2.1,
        "DCOILWTICO": 78.0,
        "DTWEXBGS": 120.0,
        "NFCI": -0.5,
    }
    macro_rows = []
    for sid, b in base.items():
        vals = b * (1 + 0.003) ** np.arange(len(months))  # gentle upward drift
        macro_rows.append(
            pd.DataFrame(
                {
                    "series_id": sid,
                    "date": months,
                    "value": vals,
                    "change": np.concatenate([[0.0], np.diff(vals)]),
                }
            )
        )
    macro = pd.concat(macro_rows, ignore_index=True)
    con.register("_m", macro)
    con.execute("create table marts.fct_macro_indicator as select * from _m")
    con.unregister("_m")
    return con


# ---------------------------------------------------------------------------
# gather_facts — macro / assets / correlations
# ---------------------------------------------------------------------------


def test_gather_facts_builds_macro_assets_correlations():
    con = _rich_con()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    assert {"as_of", "data_date", "macro", "assets", "correlations"} <= set(facts)
    macro_ids = {m["series_id"] for m in facts["macro"]}
    assert {"VIXCLS", "DGS10", "CPIAUCSL"} <= macro_ids
    # CPI is rendered as a YoY change, and there's enough history to compute it here.
    cpi = next(m for m in facts["macro"] if m["series_id"] == "CPIAUCSL")
    assert cpi["units"] == "yoy" and cpi["yoy"] is not None
    # Per-asset signals carry the momentum / MA fields.
    spy = next(a for a in facts["assets"] if a["symbol"] == "SPY")
    for k in ("ret_1d", "ret_5d", "ret_20d", "vs_ma50", "vol_20d", "asset_class"):
        assert k in spy
    # 60 days of returns across 4 assets → a real correlation read.
    assert facts["correlations"], "expected correlation notes from a 60-day window"


def test_gather_facts_has_no_portfolio_or_ml_facts():
    """The redesigned brief must NOT gather portfolio / ML / vol-skill facts."""
    con = _rich_con()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    for forbidden in ("portfolio", "portfolio_pairs", "ml_gate", "vol_skill"):
        assert forbidden not in facts


def test_gather_facts_degrades_on_empty_marts():
    """No tables → only as_of/data_date, no crash, contract still satisfied."""
    con = _make_minimal_con()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    assert set(facts) <= narrative._FACTS_REQUIRED_KEYS
    assert "macro" not in facts and "assets" not in facts


# ---------------------------------------------------------------------------
# Prompt — pre-formatted figures, narrative scope (no strategy/ML language)
# ---------------------------------------------------------------------------


def test_prompt_is_preformatted_and_scoped_to_macro_markets():
    con = _rich_con()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    block = narrative._fmt_facts_for_prompt(facts)
    assert "Macro backdrop" in block
    assert "Cross-asset moves" in block
    assert "co-movement" in block
    assert "%" in block  # returns/yields rendered as percentages
    # No raw long-float artifacts (5+ decimal places) leaked into the prompt.
    assert not re.search(r"\d\.\d{5,}", block), f"raw float leaked:\n{block}"
    # The prompt must not drag in portfolio / ML language.
    low = block.lower()
    for banned in ("sharpe", "portfolio", "strategy", "mvo", "out-of-sample", "backtest"):
        assert banned not in low, f"{banned!r} should not appear in the macro/markets prompt"


def test_system_prompt_forbids_strategy_and_ml_content():
    s = narrative._SYSTEM.lower()
    assert "do not mention portfolio" in s
    assert "sharpe" in s  # it is named in the prohibition
    assert "momentum" in s and "mean-reversion" in s and "rotation" in s


def test_prompt_handles_missing_and_nan_macro_values():
    """A YoY series without enough history is skipped; a NaN value never emits 'nan'."""
    facts = {
        "data_date": "2026-06-26",
        "macro": [
            {"series_id": "CPIAUCSL", "label": "CPI inflation", "units": "yoy", "yoy": None},
            {"series_id": "DGS10", "label": "10Y", "units": "%", "value": 4.40, "change": -0.01},
            {
                "series_id": "X",
                "label": "Broken",
                "units": "index",
                "value": float("nan"),
                "change": None,
            },
        ],
        "assets": [
            {
                "symbol": "SPY",
                "label": "S&P 500 (SPY)",
                "asset_class": "equities",
                "ret_1d": float("nan"),
                "ret_5d": 0.012,
                "ret_20d": None,
                "vs_ma50": 0.02,
                "vol_20d": 0.01,
            },
        ],
    }
    block = narrative._fmt_facts_for_prompt(facts)
    assert "nan" not in block.lower()
    assert "4.40%" in block  # the renderable macro line survives
    assert "CPI inflation" not in block  # YoY with no data is dropped
    assert "+1.2% 5d" in block or "5d +1.2%" in block


# ---------------------------------------------------------------------------
# Offline template — deterministic macro/markets report, no strategy/ML
# ---------------------------------------------------------------------------


def test_offline_brief_reports_sections_and_states_reason():
    con = _rich_con()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    brief = narrative._offline_brief(facts)
    assert "Macro backdrop" in brief
    assert "Cross-asset moves" in brief
    assert "_(deterministic template — no LLM key set)_" in brief
    low = brief.lower()
    for banned in ("sharpe", "portfolio", "strategy", "mvo"):
        assert banned not in low


def test_offline_brief_header_states_failure_reason():
    facts = {"data_date": "2026-06-25"}
    assert "data as of 2026-06-25" in narrative._offline_brief(facts)
    failed = narrative._offline_brief(facts, note="LLM temporarily unavailable")
    assert "(deterministic template — LLM temporarily unavailable)" in failed


def test_offline_brief_deterministic_byte_identical():
    con = _rich_con()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    assert narrative._offline_brief(facts) == narrative._offline_brief(facts)


# ---------------------------------------------------------------------------
# LLM output validation
# ---------------------------------------------------------------------------


def test_validate_llm_output_rejects_empty():
    assert narrative._validate_llm_output("") is not None
    assert narrative._validate_llm_output("   \n\t  ") is not None


def test_validate_llm_output_rejects_too_long():
    result = narrative._validate_llm_output("a" * (narrative._MAX_BRIEF_CHARS + 1))
    assert result is not None and "too long" in result


def test_validate_llm_output_rejects_api_key_token():
    assert narrative._validate_llm_output("Here is your api_key=SECRETVALUE summary.") is not None


def test_validate_llm_output_rejects_bearer_token():
    assert narrative._validate_llm_output("Authorization: bearer MYTOKEN12345ABC") is not None


def test_validate_llm_output_rejects_google_api_key():
    google_key = "AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    assert narrative._validate_llm_output(f"The key is {google_key}.") is not None


def test_validate_llm_output_rejects_long_hex():
    assert narrative._validate_llm_output(f"secret: {'a' * 32}") is not None


def test_validate_llm_output_accepts_normal_brief():
    normal = (
        "Equities and the dollar rose together while Treasuries slipped; the 10Y-2Y curve held at "
        "+0.30pp. VIX eased to 18.0. Watch: incoming CPI."
    )
    assert narrative._validate_llm_output(normal) is None


# ---------------------------------------------------------------------------
# generate_brief — fallback / rejection / redaction / engine tags
# ---------------------------------------------------------------------------


def test_generate_brief_falls_back_on_api_error(monkeypatch, tmp_path, caplog):
    """A live LLM failure falls back to the offline template, tags it, and redacts the key."""
    con = _rich_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    leaked = (
        "503 Server Error for url "
        "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key=AIzaSECRET123"
    )
    monkeypatch.setattr(narrative.llm, "available", lambda: True)

    def _boom(*_a, **_k):
        raise RuntimeError(leaked)

    monkeypatch.setattr(narrative.llm, "complete", _boom)
    with caplog.at_level(logging.WARNING):
        text = narrative.generate_brief(con)
    try:
        assert "_(deterministic template —" in text
        engine = con.execute(
            "select engine from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert engine == "offline-template (llm-failed)"
        assert "falling back to offline template" in caplog.text
        assert "AIzaSECRET123" not in caplog.text
        assert "key=***" in caplog.text
    finally:
        con.close()


def test_generate_brief_rejects_empty_llm_output(monkeypatch, tmp_path, caplog):
    con = _make_minimal_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    monkeypatch.setattr(narrative.llm, "available", lambda: True)
    monkeypatch.setattr(narrative.llm, "complete", lambda *_a, **_k: "")
    with caplog.at_level(logging.WARNING):
        text = narrative.generate_brief(con)
    try:
        assert "_(deterministic template —" in text
        engine = con.execute(
            "select engine from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert engine == "offline-template (llm-rejected)"
    finally:
        con.close()


def test_generate_brief_rejects_key_shaped_llm_output(monkeypatch, tmp_path, caplog):
    con = _make_minimal_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    monkeypatch.setattr(narrative.llm, "available", lambda: True)
    monkeypatch.setattr(
        narrative.llm, "complete", lambda *_a, **_k: "Summary. api_key=SUPER_SECRET_XYZ here."
    )
    with caplog.at_level(logging.WARNING):
        text = narrative.generate_brief(con)
    try:
        assert "_(deterministic template —" in text
        engine = con.execute(
            "select engine from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert engine == "offline-template (llm-rejected)"
    finally:
        con.close()


def test_generate_brief_redacts_body_in_mart_and_md_file(monkeypatch, tmp_path):
    con = _make_minimal_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    tainted = "Macro analysis. bearer FAKETOKEN123 Watch macro."
    monkeypatch.setattr(narrative, "_offline_brief", lambda _facts: tainted)
    monkeypatch.setattr(narrative.llm, "available", lambda: False)
    text = narrative.generate_brief(con)
    try:
        assert "FAKETOKEN123" not in text and "bearer ***" in text
        mart_body = con.execute(
            "select brief from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert "FAKETOKEN123" not in mart_body and "bearer ***" in mart_body
        md_files = list((tmp_path / "briefs").glob("*.md"))
        assert md_files
        md = md_files[0].read_text(encoding="utf-8")
        assert "FAKETOKEN123" not in md and "bearer ***" in md
    finally:
        con.close()


def test_generate_brief_http_429_degrades_not_crash(monkeypatch, tmp_path, caplog):
    con = _make_minimal_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    monkeypatch.setattr(narrative.llm, "available", lambda: True)

    def _rate_limited(*_a, **_k):
        request = httpx.Request("POST", "https://example.com/llm?key=SECRETKEY")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError(
            "429 for url ?key=SECRETKEY", request=request, response=response
        )

    monkeypatch.setattr(narrative.llm, "complete", _rate_limited)
    with caplog.at_level(logging.WARNING):
        text = narrative.generate_brief(con)
    try:
        assert "_(deterministic template —" in text
        engine = con.execute(
            "select engine from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert engine == "offline-template (llm-failed)"
        assert "SECRETKEY" not in caplog.text
    finally:
        con.close()


def test_engine_tags_are_distinct():
    tags = ["offline-template", "offline-template (llm-failed)", "offline-template (llm-rejected)"]
    assert len(set(tags)) == 3
    assert "(" not in "offline-template"


# ---------------------------------------------------------------------------
# gather_facts() TypedDict key-set contract
# ---------------------------------------------------------------------------


def test_gather_facts_no_extra_keys():
    con = _make_minimal_con()
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    assert not (set(facts.keys()) - narrative._FACTS_REQUIRED_KEYS)


def test_validate_facts_keys_rejects_unexpected_key():
    import pytest

    with pytest.raises(ValueError, match="unexpected keys"):
        narrative._validate_facts_keys({"as_of": "x", "UNKNOWN_KEY": 1})


def test_validate_facts_keys_accepts_all_valid_keys():
    narrative._validate_facts_keys(
        {"as_of": "x", "data_date": "x", "macro": [], "assets": [], "correlations": []}
    )
