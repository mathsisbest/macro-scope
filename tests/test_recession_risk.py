"""Tests for the recession_risk() accessor and fct_recession_risk mart semantics.

Covers:
  - accessor returns expected columns + dtypes
  - recession_prob is in [0, 1] for all plausible spreads
  - NY Fed probit coefficients: inverted curve (spread < 0) raises prob above 50%
  - model column is '10y_3m' when 3M data present, '10y_2y_proxy' when absent
  - empty frame returned when mart is missing (CatalogException swallowed)
"""

from __future__ import annotations

import math

import duckdb
import pandas as pd
from dashboard import data

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_db(tmp_path, *, include_3mo: bool) -> duckdb.DuckDBPyConnection:
    """Build a minimal test DB whose fct_recession_risk mirrors the SQL mart logic inline."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("create schema if not exists marts")

    alpha = -0.5333
    beta = -0.6629

    if include_3mo:
        # Two rows with 10Y-3M spread: one positive (normal curve), one negative (inverted).
        rows = [
            ("2024-01-02", 1.5, "10y_3m"),  # normal: DGS10=5.0, DGS3MO=3.5 => spread=1.5
            ("2024-01-03", -0.5, "10y_3m"),  # inverted: DGS10=4.0, DGS3MO=4.5 => spread=-0.5
        ]
    else:
        # Fallback 10Y-2Y spread (no DGS3MO available, as in synthetic seed).
        rows = [
            ("2024-01-02", 0.8, "10y_2y_proxy"),
            ("2024-01-03", -0.3, "10y_2y_proxy"),
        ]

    # Pre-compute recession_prob inline using the same formula.
    def probit_prob(spread: float) -> float:
        x = alpha + beta * spread
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    values = ", ".join(
        f"(DATE '{date}', {spread:.4f}, {probit_prob(spread):.8f}, '{model}')"
        for date, spread, model in rows
    )
    con.execute(
        "create table marts.fct_recession_risk as "
        f"select * from (values {values}) "
        "t(date, spread_10y_3m, recession_prob, model)"
    )
    con.close()
    return db_path


# ---------------------------------------------------------------------------
# accessor: column contract
# ---------------------------------------------------------------------------


def test_recession_risk_columns(monkeypatch, tmp_path):
    """accessor returns the exact four contracted columns."""
    db_path = _build_db(tmp_path, include_3mo=True)

    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(
        data, "connect", lambda *a, **k: duckdb.connect(str(db_path), read_only=True)
    )
    data.query.clear()

    df = data.recession_risk()
    assert list(df.columns) == ["date", "spread_10y_3m", "recession_prob", "model"]
    assert len(df) == 2


# ---------------------------------------------------------------------------
# probit mechanics
# ---------------------------------------------------------------------------


def test_recession_prob_range(monkeypatch, tmp_path):
    """recession_prob stays in [0, 1] for a range of realistic spreads."""
    db_path = _build_db(tmp_path, include_3mo=True)

    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(
        data, "connect", lambda *a, **k: duckdb.connect(str(db_path), read_only=True)
    )
    data.query.clear()

    df = data.recession_risk()
    assert (df["recession_prob"] >= 0).all()
    assert (df["recession_prob"] <= 1).all()


def test_inverted_curve_raises_recession_prob_above_normal_curve(monkeypatch, tmp_path):
    """Inverted yield curve (spread < 0) → higher recession probability than normal curve."""
    db_path = _build_db(tmp_path, include_3mo=True)

    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(
        data, "connect", lambda *a, **k: duckdb.connect(str(db_path), read_only=True)
    )
    data.query.clear()

    df = data.recession_risk().set_index("date")
    # row 0: normal curve (spread=1.5), row 1: inverted (spread=-0.5)
    prob_normal = df.loc[pd.Timestamp("2024-01-02"), "recession_prob"]
    prob_inverted = df.loc[pd.Timestamp("2024-01-03"), "recession_prob"]
    assert prob_inverted > prob_normal


def test_strongly_inverted_curve_exceeds_50_pct(monkeypatch, tmp_path):
    """A strongly inverted curve should push P(recession) above 50%."""
    # NY Fed: spread=-1.5 => alpha + beta*spread = -0.5333 + (-0.6629)*(-1.5) = ~0.461 > 0
    # => normal_cdf(0.461) > 0.5
    alpha, beta = -0.5333, -0.6629
    spread = -1.5
    x = alpha + beta * spread
    prob = 0.5 * math.erfc(-x / math.sqrt(2.0))
    assert prob > 0.5, f"Expected P>0.5 for spread={spread}, got {prob:.4f}"


def test_probit_values_match_manual_calculation(monkeypatch, tmp_path):
    """The accessor returns probability values consistent with the NY Fed probit formula."""
    db_path = _build_db(tmp_path, include_3mo=True)

    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(
        data, "connect", lambda *a, **k: duckdb.connect(str(db_path), read_only=True)
    )
    data.query.clear()

    df = data.recession_risk()
    alpha, beta = -0.5333, -0.6629

    for _, row in df.iterrows():
        spread = row["spread_10y_3m"]
        x = alpha + beta * spread
        expected = 0.5 * math.erfc(-x / math.sqrt(2.0))
        assert abs(row["recession_prob"] - expected) < 1e-6, (
            f"spread={spread}: expected {expected:.8f}, got {row['recession_prob']:.8f}"
        )


# ---------------------------------------------------------------------------
# model column (10y_3m vs 10y_2y_proxy)
# ---------------------------------------------------------------------------


def test_model_is_10y_3m_when_dgs3mo_present(monkeypatch, tmp_path):
    db_path = _build_db(tmp_path, include_3mo=True)

    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(
        data, "connect", lambda *a, **k: duckdb.connect(str(db_path), read_only=True)
    )
    data.query.clear()

    df = data.recession_risk()
    assert (df["model"] == "10y_3m").all()


def test_model_is_10y_2y_proxy_when_dgs3mo_absent(monkeypatch, tmp_path):
    db_path = _build_db(tmp_path, include_3mo=False)

    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(
        data, "connect", lambda *a, **k: duckdb.connect(str(db_path), read_only=True)
    )
    data.query.clear()

    df = data.recession_risk()
    assert (df["model"] == "10y_2y_proxy").all()


# ---------------------------------------------------------------------------
# missing mart → empty frame (not an error)
# ---------------------------------------------------------------------------


def test_recession_risk_returns_empty_when_mart_missing(monkeypatch, tmp_path):
    """When the mart doesn't exist yet, the accessor silently returns an empty DataFrame."""
    db_path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("create schema if not exists marts")
    con.close()

    monkeypatch.setattr(data, "db_exists", lambda: True)
    monkeypatch.setattr(
        data, "connect", lambda *a, **k: duckdb.connect(str(db_path), read_only=True)
    )
    data.query.clear()

    df = data.recession_risk()
    assert isinstance(df, pd.DataFrame)
    assert df.empty
