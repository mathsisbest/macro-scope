"""The market brief is numerically grounded: portfolio stats + CIs come from the marts, and the
brief hedges (says "not distinguishable") when the bootstrap says so — never invented.

Also covers:
- post-generation LLM output validation (empty / too-long / key-shaped → llm-rejected tag)
- body redaction: redact() runs on EVERY persisted body before .md write + mart insert
- GC: deterministic ordering → byte-identical offline brief regardless of row order
- GC: HTTP 429 transport error → 'offline-template (llm-failed)', not a crash; tags distinct
- GC: gather_facts() TypedDict key-set contract — unexpected/missing key raises ValueError
"""

import logging

import duckdb
import httpx
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


def test_generate_brief_falls_back_to_offline_template_on_api_error(monkeypatch, tmp_path, caplog):
    """A live LLM failure must not crash the brief: it falls back to the deterministic offline
    template, persists the fallback engine tag, and the provider key (which rides in the httpx
    error URL) is redacted from the warning — never leaked to logs/CI."""
    con = _con_with_portfolio()
    # Keep the brief file write hermetic (generate_brief writes to <duckdb parent>/briefs/).
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    # Force the live path; the provider call then raises with a key-bearing URL in its message.
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
        # Fell back to the deterministic template instead of raising.
        assert "_(template; set an LLM key for AI narrative)_" in text
        # Persisted, tagged distinctly from the no-key path so a failed live call is legible.
        engine = con.execute(
            "select engine from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert engine == "offline-template (llm-failed)"
        # The fallback was logged, but the API key was scrubbed before it reached the log.
        assert "falling back to offline template" in caplog.text
        assert "AIzaSECRET123" not in caplog.text
        assert "key=***" in caplog.text
    finally:
        con.close()


# ---------------------------------------------------------------------------
# GB: LLM output validation tests
# ---------------------------------------------------------------------------


def test_validate_llm_output_rejects_empty():
    assert narrative._validate_llm_output("") is not None
    assert narrative._validate_llm_output("   \n\t  ") is not None


def test_validate_llm_output_rejects_too_long():
    long_text = "a" * (narrative._MAX_BRIEF_CHARS + 1)
    result = narrative._validate_llm_output(long_text)
    assert result is not None
    assert "too long" in result


def test_validate_llm_output_rejects_api_key_token():
    # api_key=... style
    assert narrative._validate_llm_output("Here is your api_key=SECRETVALUE summary.") is not None


def test_validate_llm_output_rejects_bearer_token():
    assert narrative._validate_llm_output("Authorization: bearer MYTOKEN12345ABC") is not None


def test_validate_llm_output_rejects_google_api_key():
    google_key = "AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    assert narrative._validate_llm_output(f"The key is {google_key}.") is not None


def test_validate_llm_output_rejects_long_hex():
    hex_secret = "a" * 32  # 32-char hex-alphabet string
    assert narrative._validate_llm_output(f"secret: {hex_secret}") is not None


def test_validate_llm_output_accepts_normal_brief():
    normal = (
        "SPY closed at $450.23 (+0.3%). The 10Y-2Y spread is -0.15pp. "
        "Risk parity returned +12.5% with Sharpe 1.10 [0.50, 1.70]. "
        "Watch: upcoming CPI release."
    )
    assert narrative._validate_llm_output(normal) is None


# ---------------------------------------------------------------------------
# GB: generate_brief → llm-rejected tag when LLM output fails validation
# ---------------------------------------------------------------------------


def _make_minimal_con() -> duckdb.DuckDBPyConnection:
    """Minimal in-memory DB with just the marts schema (no portfolio tables)."""
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    return con


def test_generate_brief_rejects_empty_llm_output(monkeypatch, tmp_path, caplog):
    """Empty LLM output → offline-template (llm-rejected) tag, not a crash."""
    con = _make_minimal_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    monkeypatch.setattr(narrative.llm, "available", lambda: True)
    monkeypatch.setattr(narrative.llm, "complete", lambda *_a, **_k: "")

    with caplog.at_level(logging.WARNING):
        text = narrative.generate_brief(con)

    try:
        assert "_(template; set an LLM key for AI narrative)_" in text
        engine = con.execute(
            "select engine from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert engine == "offline-template (llm-rejected)"
        assert "falling back to offline template" in caplog.text
    finally:
        con.close()


def test_generate_brief_rejects_too_long_llm_output(monkeypatch, tmp_path, caplog):
    """Over-length LLM output → offline-template (llm-rejected)."""
    con = _make_minimal_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    monkeypatch.setattr(narrative.llm, "available", lambda: True)
    too_long = "x " * (narrative._MAX_BRIEF_CHARS + 1)
    monkeypatch.setattr(narrative.llm, "complete", lambda *_a, **_k: too_long)

    with caplog.at_level(logging.WARNING):
        text = narrative.generate_brief(con)

    try:
        assert "_(template; set an LLM key for AI narrative)_" in text
        engine = con.execute(
            "select engine from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert engine == "offline-template (llm-rejected)"
    finally:
        con.close()


def test_generate_brief_rejects_key_shaped_llm_output(monkeypatch, tmp_path, caplog):
    """LLM output containing a key-shaped token → offline-template (llm-rejected)."""
    con = _make_minimal_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    monkeypatch.setattr(narrative.llm, "available", lambda: True)
    # Simulate an LLM that accidentally echoes back a credential-like string.
    key_bearing = "Here is the summary. api_key=SUPER_SECRET_VALUE_XYZ for reference."
    monkeypatch.setattr(narrative.llm, "complete", lambda *_a, **_k: key_bearing)

    with caplog.at_level(logging.WARNING):
        text = narrative.generate_brief(con)

    try:
        assert "_(template; set an LLM key for AI narrative)_" in text
        engine = con.execute(
            "select engine from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert engine == "offline-template (llm-rejected)"
        assert "falling back to offline template" in caplog.text
    finally:
        con.close()


# ---------------------------------------------------------------------------
# C7: vol_skill escape-hatch — NOT cleared → no-edge sentence; cleared → positive allowed
# ---------------------------------------------------------------------------


def _make_model_metrics_df(cleared: bool) -> "pd.DataFrame":
    """Return a minimal model_metrics long-format DataFrame.

    cleared=False: oos_r2 below threshold (0.05 < 0.10).
    cleared=True:  oos_r2 above threshold (0.50 >= 0.10) + qlike_skill_ratio < 0.99 + folds ok.
    """
    if cleared:
        rows = [
            {"model": "rv_har", "symbol": "SPY", "metric": "oos_r2", "value": 0.50},
            {"model": "rv_har", "symbol": "SPY", "metric": "qlike_skill_ratio", "value": 0.90},
            {"model": "rv_har", "symbol": "SPY", "metric": "folds_passed", "value": 3},
            {"model": "rv_har", "symbol": "SPY", "metric": "n_folds", "value": 5},
            {"model": "rv_har", "symbol": "SPY", "metric": "n_obs", "value": 300},
        ]
    else:
        rows = [
            {"model": "rv_har", "symbol": "SPY", "metric": "oos_r2", "value": 0.05},
            {"model": "rv_har", "symbol": "SPY", "metric": "qlike_skill_ratio", "value": 1.10},
            {"model": "rv_har", "symbol": "SPY", "metric": "folds_passed", "value": 1},
            {"model": "rv_har", "symbol": "SPY", "metric": "n_folds", "value": 5},
            {"model": "rv_har", "symbol": "SPY", "metric": "n_obs", "value": 300},
        ]
    return pd.DataFrame(rows)


def test_offline_brief_not_cleared_no_edge_sentence_no_beats():
    """When vol_skill is NOT cleared, the brief must contain the no-edge sentence
    and must NOT contain 'beats' or 'outperforms'."""
    facts = {
        "as_of": "2024-01-02 00:00 UTC",
        "vol_skill": {"cleared": False, "oos_r2": 0.05, "reasons": ["oos_r2=0.05 < R2_MIN=0.10"]},
    }
    brief = narrative._offline_brief(facts)
    # Must state clearly there is no edge
    no_edge = (
        "no demonstrated out-of-sample skill" in brief or "no reliable out-of-sample edge" in brief
    )
    assert no_edge
    # Must NOT use positive phrasing
    assert "beats" not in brief.lower()
    assert "outperforms" not in brief.lower()
    assert "beat its" not in brief.lower()


def test_offline_brief_cleared_allows_positive_phrasing():
    """When vol_skill IS cleared, the brief may say the model beat its baseline."""
    facts = {
        "as_of": "2024-01-02 00:00 UTC",
        "vol_skill": {"cleared": True, "oos_r2": 0.50, "reasons": []},
    }
    brief = narrative._offline_brief(facts)
    assert "beat its persistence baseline" in brief


def test_gather_facts_includes_vol_skill_not_cleared(tmp_path):
    """gather_facts sources skill_verdict() and stores vol_skill['cleared']=False
    when model_metrics has below-threshold metrics."""
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    metrics_df = _make_model_metrics_df(cleared=False)
    con.register("_m", metrics_df)
    con.execute("create table marts.model_metrics as select * from _m")
    con.unregister("_m")
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    assert "vol_skill" in facts
    assert facts["vol_skill"]["cleared"] is False


def test_gather_facts_includes_vol_skill_cleared(tmp_path):
    """gather_facts sources skill_verdict() and stores vol_skill['cleared']=True
    when model_metrics has above-threshold metrics."""
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    metrics_df = _make_model_metrics_df(cleared=True)
    con.register("_m", metrics_df)
    con.execute("create table marts.model_metrics as select * from _m")
    con.unregister("_m")
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    assert "vol_skill" in facts
    assert facts["vol_skill"]["cleared"] is True


def test_gather_facts_vol_skill_fails_closed_on_missing_mart():
    """gather_facts must not crash when marts.model_metrics doesn't exist;
    skill_verdict() fails closed (cleared=False)."""
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    # No model_metrics table — skill_verdict will receive an empty DataFrame
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    # vol_skill must be present but cleared=False (fail-closed)
    assert "vol_skill" in facts
    assert facts["vol_skill"]["cleared"] is False


# ---------------------------------------------------------------------------
# GB: body redaction — redact() runs before BOTH the .md write and mart insert
# ---------------------------------------------------------------------------


def test_generate_brief_redacts_body_in_mart_and_md_file(monkeypatch, tmp_path):
    """A brief body containing a fake key is redacted before it reaches the mart and .md."""
    con = _make_minimal_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    # Force the offline path (no LLM key) but inject a body that contains a bearer token
    # by monkeypatching _offline_brief to return tainted content.
    tainted = "SPY analysis. bearer FAKETOKEN123 Watch macro."
    monkeypatch.setattr(narrative, "_offline_brief", lambda _facts: tainted)
    monkeypatch.setattr(narrative.llm, "available", lambda: False)

    text = narrative.generate_brief(con)

    try:
        # The returned text must have the token scrubbed.
        assert "FAKETOKEN123" not in text
        assert "bearer ***" in text

        # The mart body must be scrubbed.
        mart_body = con.execute(
            "select brief from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert "FAKETOKEN123" not in mart_body
        assert "bearer ***" in mart_body

        # The .md file must be scrubbed.
        briefs_dir = tmp_path / "briefs"
        md_files = list(briefs_dir.glob("*.md"))
        assert md_files, "expected at least one .md brief file"
        md_content = md_files[0].read_text(encoding="utf-8")
        assert "FAKETOKEN123" not in md_content
        assert "bearer ***" in md_content
    finally:
        con.close()


# ---------------------------------------------------------------------------
# GC: Deterministic ordering — same facts, different row order → byte-identical brief
# ---------------------------------------------------------------------------


def _portfolio_facts_base() -> dict:
    """Minimal facts dict with two portfolio strategies and two pairs."""
    return {
        "as_of": "2024-01-02 00:00 UTC",
        "data_date": "2024-01-02",
        "portfolio": [
            {
                "strategy": "sixty_forty",
                "total_return": 0.08,
                "max_drawdown": -0.01,
                "sharpe": 1.10,
                "sharpe_lo": 0.20,
                "sharpe_hi": 2.00,
            },
            {
                "strategy": "equal_weight",
                "total_return": 0.05,
                "max_drawdown": -0.02,
                "sharpe": 0.80,
                "sharpe_lo": 0.10,
                "sharpe_hi": 1.50,
            },
        ],
        "portfolio_pairs": [
            {"strategy_a": "risk_parity", "strategy_b": "sixty_forty", "distinguishable": True},
            {"strategy_a": "equal_weight", "strategy_b": "sixty_forty", "distinguishable": True},
        ],
    }


def test_offline_brief_deterministic_same_row_order():
    """Same facts in the same order → identical brief (sanity check)."""
    facts = _portfolio_facts_base()
    brief1 = narrative._offline_brief(facts)
    brief2 = narrative._offline_brief(facts)
    assert brief1 == brief2


def test_offline_brief_byte_identical_regardless_of_portfolio_row_order():
    """Same portfolio facts but rows in reversed order must produce a byte-identical brief."""
    facts_a = _portfolio_facts_base()
    facts_b = _portfolio_facts_base()
    # Reverse the portfolio rows
    facts_b["portfolio"] = list(reversed(facts_b["portfolio"]))

    brief_a = narrative._offline_brief(facts_a)
    brief_b = narrative._offline_brief(facts_b)
    assert brief_a == brief_b, (
        "Offline brief is NOT byte-identical when portfolio rows arrive in different order.\n"
        f"--- facts_a brief ---\n{brief_a}\n--- facts_b brief ---\n{brief_b}"
    )


def test_offline_brief_byte_identical_regardless_of_pairs_row_order():
    """Same pairs facts but rows in reversed order must produce a byte-identical brief."""
    facts_a = _portfolio_facts_base()
    facts_b = _portfolio_facts_base()
    # Reverse the portfolio_pairs rows
    facts_b["portfolio_pairs"] = list(reversed(facts_b["portfolio_pairs"]))

    brief_a = narrative._offline_brief(facts_a)
    brief_b = narrative._offline_brief(facts_b)
    assert brief_a == brief_b, (
        "Offline brief is NOT byte-identical when portfolio_pairs rows arrive in different order.\n"
        f"--- facts_a brief ---\n{brief_a}\n--- facts_b brief ---\n{brief_b}"
    )


# ---------------------------------------------------------------------------
# GC: Engine tags — HTTP 429 → 'offline-template (llm-failed)', distinct tags
# ---------------------------------------------------------------------------


def test_generate_brief_http_429_degrades_to_llm_failed_not_crash(monkeypatch, tmp_path, caplog):
    """An HTTP 429 from the LLM provider must degrade to 'offline-template (llm-failed)',
    NOT crash, and the key must not appear in the logs."""
    con = _make_minimal_con()
    monkeypatch.setattr(narrative.settings, "duckdb_path", tmp_path / "ci.duckdb")
    monkeypatch.setattr(narrative.llm, "available", lambda: True)

    def _rate_limited(*_a, **_k):
        # Simulate what httpx raises on a 429 response (HTTPStatusError is a subclass of HTTPError)
        request = httpx.Request("POST", "https://example.com/llm?key=SECRETKEY")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError(
            "429 Too Many Requests for url https://example.com/llm?key=SECRETKEY",
            request=request,
            response=response,
        )

    monkeypatch.setattr(narrative.llm, "complete", _rate_limited)

    with caplog.at_level(logging.WARNING):
        text = narrative.generate_brief(con)

    try:
        # Must not crash — fell back to the offline template.
        assert "_(template; set an LLM key for AI narrative)_" in text
        # Engine tag must be 'offline-template (llm-failed)', not the no-key variant.
        engine = con.execute(
            "select engine from marts.market_brief order by created_at desc limit 1"
        ).fetchone()[0]
        assert engine == "offline-template (llm-failed)"
        # The warning must be present and the key must be redacted.
        assert "falling back to offline template" in caplog.text
        assert "SECRETKEY" not in caplog.text
    finally:
        con.close()


def test_engine_tags_are_distinct():
    """The three offline engine tags must all be distinct strings."""
    tags = [
        "offline-template",
        "offline-template (llm-failed)",
        "offline-template (llm-rejected)",
    ]
    assert len(set(tags)) == 3, "Engine tags are not distinct — a branch will be undiagnosable"
    # The no-key tag must NOT contain a parenthetical suffix.
    assert "(" not in "offline-template"
    # The failed tag signals a transport error (429, 5xx, network).
    assert "llm-failed" in "offline-template (llm-failed)"
    # The rejected tag signals the LLM responded but the output failed validation.
    assert "llm-rejected" in "offline-template (llm-rejected)"


# ---------------------------------------------------------------------------
# GC: gather_facts() TypedDict key-set contract
# ---------------------------------------------------------------------------


def test_gather_facts_typeddict_key_set_no_extra_keys():
    """gather_facts() must return ONLY keys defined in FactsDict — no undocumented extras."""
    con = duckdb.connect()
    con.execute("create schema if not exists marts")
    try:
        facts = narrative.gather_facts(con)
    finally:
        con.close()
    allowed = narrative._FACTS_REQUIRED_KEYS
    extra = set(facts.keys()) - allowed
    assert not extra, f"gather_facts() returned undocumented keys: {extra!r}"


def test_validate_facts_keys_rejects_unexpected_key():
    """_validate_facts_keys() must raise ValueError on an unknown key."""
    bad_facts = {
        "as_of": "2024-01-02",
        "data_date": "2024-01-02",
        "UNKNOWN_KEY": "should not be here",
    }
    import pytest

    with pytest.raises(ValueError, match="unexpected keys"):
        narrative._validate_facts_keys(bad_facts)


def test_validate_facts_keys_accepts_all_valid_keys():
    """_validate_facts_keys() must not raise when all keys are in the FactsDict contract."""
    valid_facts = {
        "as_of": "2024-01-02",
        "data_date": "2024-01-02",
        "crypto": [],
        "spy": {},
        "yields": {},
        "portfolio": [],
        "portfolio_pairs": [],
        "ml_gate": {},
        "vol_skill": {},
    }
    # Must not raise
    narrative._validate_facts_keys(valid_facts)


def test_validate_facts_keys_accepts_partial_keys():
    """_validate_facts_keys() must not raise when only some optional keys are present."""
    partial_facts = {"as_of": "2024-01-02", "data_date": "2024-01-02"}
    narrative._validate_facts_keys(partial_facts)
