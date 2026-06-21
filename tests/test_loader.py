import pandas as pd

from mmi.ingestion.loader import DuckDBLoader


def test_upsert_is_idempotent_and_updates(con):
    loader = DuckDBLoader(con)
    df = pd.DataFrame({"symbol": ["A", "B"], "date": [1, 2], "v": [10, 20]})

    loader.upsert("raw.t", df, ["symbol", "date"])
    loader.upsert("raw.t", df, ["symbol", "date"])  # re-run must not duplicate
    assert con.execute("select count(*) from raw.t").fetchone()[0] == 2

    updated = pd.DataFrame({"symbol": ["A"], "date": [1], "v": [99]})
    loader.upsert("raw.t", updated, ["symbol", "date"])
    assert con.execute("select v from raw.t where symbol='A' and date=1").fetchone()[0] == 99
    assert con.execute("select count(*) from raw.t").fetchone()[0] == 2


def test_audit_log_written(con):
    loader = DuckDBLoader(con)
    run_id = loader.start_run("unit")
    loader.finish_run(run_id, 5, "success")
    row = con.execute(
        "select rows, status from raw.pipeline_runs where run_id = ?", [run_id]
    ).fetchone()
    assert row == (5, "success")
