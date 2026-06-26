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


def test_range_start_presets_map_to_correct_floor():
    """range_start() turns a Google-style preset into an ISO date floor relative to the anchor
    (the latest data date). 'Max'/None/empty/unknown -> None (no filter)."""
    anchor = "2026-06-25"
    assert data.range_start("Max", anchor) is None
    assert data.range_start(None, anchor) is None
    assert data.range_start("1M", anchor) == "2026-05-26"  # 30 days back
    assert data.range_start("6M", anchor) == "2025-12-25"  # 182 days back
    assert data.range_start("1Y", anchor) == "2025-06-25"  # 365 days back
    assert data.range_start("5Y", anchor) == "2021-06-26"  # 5*365 days back
    assert data.range_start("YTD", anchor) == "2026-01-01"  # Jan 1 of the anchor's year
    # No anchor or an unparseable/unknown value -> no floor (don't crash).
    assert data.range_start("1Y", "") is None
    assert data.range_start("bogus", anchor) is None
    assert data.range_start("1M", "not-a-date") is None


def test_asset_daily_start_filters_rows(monkeypatch, tmp_path):
    """asset_daily(symbol, start) adds a `date >= ?` floor; without start returns all rows."""
    db = tmp_path / "m.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema marts")
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        "('SPY', DATE '2020-01-01', 10.0, 0.0, 1.0, 9.0), "
        "('SPY', DATE '2026-06-25', 20.0, 0.0, 2.0, 19.0)) "
        "as t(symbol, date, close, daily_return, vol_20d, ma_50)"
    )
    con.close()
    monkeypatch.setattr(settings_mod.settings, "snapshot_mode", False)
    monkeypatch.setattr(settings_mod.settings, "duckdb_path", db)
    data.query.clear()
    assert len(data.asset_daily("SPY")) == 2  # no floor → both rows
    data.query.clear()
    recent = data.asset_daily("SPY", "2026-01-01")  # floor drops the 2020 row
    assert len(recent) == 1 and str(recent["date"].iloc[0]).startswith("2026")


def test_all_assets_daily_returns_long_frame_for_every_asset_windowed(monkeypatch, tmp_path):
    """all_assets_daily(start) returns [symbol, asset_class, date, close, daily_return] for every
    asset, ordered by symbol,date and filtered by the `date >= start` floor when given."""
    db = tmp_path / "m.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create schema marts")
    con.execute(
        "create table marts.fct_asset_daily as select * from (values "
        "('SPY', 'equities', DATE '2020-01-01', 10.0, 0.0, 1.0, 9.0), "
        "('SPY', 'equities', DATE '2026-06-25', 20.0, 0.10, 2.0, 19.0), "
        "('TLT', 'bonds',    DATE '2026-06-25', 99.0, -0.01, 3.0, 98.0)) "
        "as t(symbol, asset_class, date, close, daily_return, vol_20d, ma_50)"
    )
    con.close()
    monkeypatch.setattr(settings_mod.settings, "snapshot_mode", False)
    monkeypatch.setattr(settings_mod.settings, "duckdb_path", db)

    data.query.clear()
    full = data.all_assets_daily()
    assert set(full.columns) == {"symbol", "asset_class", "date", "close", "daily_return"}
    assert len(full) == 3  # every asset's every row, no floor
    assert list(full["symbol"]) == ["SPY", "SPY", "TLT"]  # ordered by symbol, then date

    data.query.clear()
    windowed = data.all_assets_daily("2026-01-01")  # floor drops the 2020 SPY row
    assert len(windowed) == 2
    assert set(windowed["symbol"]) == {"SPY", "TLT"}


def test_macro_catalog_exposes_label_category_units():
    """macro_catalog() surfaces the config metadata the Macro tab groups + labels by."""
    cat = data.macro_catalog()
    assert cat, "macro catalogue should be non-empty"
    ids = {c["id"] for c in cat}
    assert {"CPIAUCSL", "VIXCLS", "GFDEGDQ188S"} <= ids  # existing + newly-added series
    for c in cat:
        assert c["id"] and c["label"] and c["category"]  # metadata present for every entry


def test_macro_config_entries_all_carry_metadata():
    """Every macro series in config must declare id + category + units, so the monitor can group
    and caption it. Guards future additions from dropping the metadata."""
    from mmi.settings import load_assets

    for entry in load_assets()["macro"]:
        assert {"id", "category", "units"} <= set(entry), f"macro entry missing metadata: {entry}"
