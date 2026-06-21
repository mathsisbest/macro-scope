"""End-to-end offline smoke test: seed -> marts -> ML -> GenAI (template)."""

from mmi import sampledata, transform_fallback
from mmi import settings as settings_mod
from mmi.ai.narrative import generate_brief
from mmi.ml.pipeline import run_ml


def test_full_offline_pipeline(con, monkeypatch, tmp_path):
    monkeypatch.setattr(settings_mod.settings, "duckdb_path", tmp_path / "t.duckdb")

    counts = sampledata.seed(con)
    assert counts["raw.asset_prices"] > 0

    transform_fallback.build_marts(con)
    assert con.execute("select count(*) from marts.fct_asset_daily").fetchone()[0] > 0
    con.execute("select * from marts.fct_market_macro limit 1")  # ASOF join builds

    summary = run_ml(con)
    assert any("dir_acc" in k for k in summary)

    brief = generate_brief(con)  # no LLM key -> deterministic template
    assert isinstance(brief, str) and len(brief) > 0
