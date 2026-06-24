"""db_exists() short-circuits True on MotherDuck and checks the file locally; window filtering."""

import duckdb
from dashboard import data

import mmi.settings as settings_mod


def test_db_exists_true_on_motherduck(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_mod.settings, "motherduck_database", "mmi")
    monkeypatch.setattr(settings_mod.settings, "motherduck_token", "tok")
    # No local file exists, yet MotherDuck is configured -> present.
    monkeypatch.setattr(settings_mod.settings, "duckdb_path", tmp_path / "absent.duckdb")
    assert data.db_exists() is True


def test_db_exists_false_when_local_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_mod.settings, "motherduck_database", "")
    monkeypatch.setattr(settings_mod.settings, "motherduck_token", "")
    monkeypatch.setattr(settings_mod.settings, "duckdb_path", tmp_path / "absent.duckdb")
    assert data.db_exists() is False


def test_portfolio_windows_ordered_and_accessor_filters_by_window(monkeypatch, tmp_path):
    db = tmp_path / "m.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists marts")
    # Two windows present, deliberately inserted out of canonical order.
    con.execute(
        "create table marts.fct_portfolio_returns as select * from (values "
        "('inc_btc_2015', 'equal_weight', DATE '2020-01-02', 0.01, 0.01, 0.0, 1.0), "
        "('ex_btc_2002', 'equal_weight', DATE '2010-01-04', 0.02, 0.02, 0.0, 1.0)) "
        "as t(window_id, strategy, date, daily_return, cumulative_return, drawdown, "
        "rolling_sharpe_252)"
    )
    con.close()
    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(data, "connect", lambda *a, **k: duckdb.connect(str(db), read_only=True))
    data.query.clear()  # st.cache_data — avoid cross-test contamination

    # present windows come back in canonical enum order, not insertion order
    assert data.portfolio_windows() == ["ex_btc_2002", "inc_btc_2015"]
    # the accessor returns ONLY the requested window's rows (the single-window choke point)
    inc = data.portfolio_returns("inc_btc_2015")
    assert len(inc) == 1 and inc["cumulative_return"].iloc[0] == 0.01
    ex = data.portfolio_returns("ex_btc_2002")
    assert len(ex) == 1 and ex["cumulative_return"].iloc[0] == 0.02


def _provenance_db(tmp_path, source: str) -> "duckdb.DuckDBPyConnection":
    db = tmp_path / "m.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists marts")
    con.execute(
        "create table marts.fct_market_macro as select * from (values "
        "(DATE '2026-06-22', 1.0), (DATE '2026-06-23', 2.0)) t(date, spy_close)"
    )
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        f"('SPY', DATE '2026-06-23', '{source}')) t(symbol, date, source)"
    )
    con.close()
    return db


def test_data_as_of_is_max_market_date_and_sample_flag_true(monkeypatch, tmp_path):
    db = _provenance_db(tmp_path, "sample")
    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(data, "connect", lambda *a, **k: duckdb.connect(str(db), read_only=True))
    data.query.clear()

    assert data.data_as_of() == "2026-06-23"  # max(date), ISO string
    assert data.is_sample_data() is True  # source is only "sample"


def test_is_sample_data_false_for_real_source(monkeypatch, tmp_path):
    db = _provenance_db(tmp_path, "yahoo")
    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(data, "connect", lambda *a, **k: duckdb.connect(str(db), read_only=True))
    data.query.clear()

    assert data.is_sample_data() is False  # a real ingestion source -> live


def test_is_sample_data_none_when_mixed_or_null_source(monkeypatch, tmp_path):
    # A partial ingest: one symbol still synthetic, one live, plus an unrecorded (NULL) source.
    db = tmp_path / "m.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists marts")
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        "('SPY', DATE '2026-06-23', 'sample'), "
        "('VEA', DATE '2026-06-23', 'yahoo'), "
        "('TLT', DATE '2026-06-23', CAST(NULL AS VARCHAR))) t(symbol, date, source)"
    )
    con.close()
    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(data, "connect", lambda *a, **k: duckdb.connect(str(db), read_only=True))
    data.query.clear()

    # Mixed sample+live must NOT be reported as live — the provenance is ambiguous.
    assert data.is_sample_data() is None


def test_is_sample_data_none_when_only_null_source(monkeypatch, tmp_path):
    db = tmp_path / "m.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists marts")
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        "('SPY', DATE '2026-06-23', CAST(NULL AS VARCHAR))) t(symbol, date, source)"
    )
    con.close()
    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(data, "connect", lambda *a, **k: duckdb.connect(str(db), read_only=True))
    data.query.clear()

    assert data.is_sample_data() is None  # no usable provenance signal


def test_provenance_degrades_when_marts_absent(monkeypatch, tmp_path):
    db = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists marts")
    con.close()
    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(data, "connect", lambda *a, **k: duckdb.connect(str(db), read_only=True))
    data.query.clear()

    assert data.data_as_of() == ""  # missing mart -> empty frame -> ""
    assert data.is_sample_data() is None  # missing mart -> empty frame -> None


def test_is_sample_data_none_when_live_plus_null_source(monkeypatch, tmp_path):
    # One live row + one unrecorded (NULL) row: NOT 'all live'. The invariant says don't claim it.
    db = tmp_path / "m.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema if not exists marts")
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        "('SPY', DATE '2026-06-23', 'yahoo'), "
        "('VEA', DATE '2026-06-23', CAST(NULL AS VARCHAR))) t(symbol, date, source)"
    )
    con.close()
    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(data, "connect", lambda *a, **k: duckdb.connect(str(db), read_only=True))
    data.query.clear()

    assert data.is_sample_data() is None  # an unrecorded source must not read as live


def test_macro_source_caption_is_honest_per_provenance():
    # Live FRED data earns the FRED attribution.
    live = data.macro_source_caption(False)
    assert live.startswith("Source: FRED, Federal Reserve Bank of St. Louis")
    # Sample data must be flagged synthetic and must NOT be attributed to FRED.
    sample = data.macro_source_caption(True)
    assert "Source: FRED" not in sample and "not from FRED" in sample
    # Mixed / unknown provenance makes no source claim at all.
    assert data.macro_source_caption(None) == ""
