"""db_exists() short-circuits True on MotherDuck and checks the file locally."""

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
