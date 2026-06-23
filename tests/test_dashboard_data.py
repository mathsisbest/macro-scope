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
