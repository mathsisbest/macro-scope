"""Regression tests for the per-symbol ML config default-inheritance merge (P2-2.4).

`_SYMBOL_ML_CONFIG` now holds only per-symbol *overrides*; `_ml_config` merges them
over `_DEFAULT_ML_CONFIG`. The resolved config for every symbol must be byte-for-byte
identical to the pre-refactor hard-coded values (no model params, feature sets, or
training behaviour changed).
"""

from __future__ import annotations

from mmi.ml.pipeline import _DEFAULT_ML_CONFIG, _SYMBOL_ML_CONFIG, _ml_config

# Resolved (post-merge) config for every configured symbol, locked to the
# pre-refactor values from the full per-symbol dicts.
_EXPECTED_RESOLVED: dict[str, dict] = {
    "SPY": {
        "model": "gb",
        "train_size": 2520,
        "target_horizon": 20,
        "use_all_train": True,
        "feature_set": "vol_rich_plus",
    },
    "QQQ": {
        "model": "gb",
        "train_size": 1260,
        "target_horizon": 20,
        "use_all_train": True,
        "feature_set": "vol_rich_plus",
    },
    "GLD": {
        "model": "gb",
        "train_size": 1512,
        "target_horizon": 20,
        "use_all_train": True,
        "feature_set": "vol_rich_plus",
    },
    "TLT": {
        "model": "lgb",
        "train_size": 1764,
        "target_horizon": 20,
        "use_all_train": True,
        "feature_set": "vol_rich_plus",
    },
    "BTC": {
        "model": "gb",
        "train_size": 1008,
        "target_horizon": 20,
        "use_all_train": True,
        "feature_set": "vol_rich_plus",
    },
}


def test_resolved_config_matches_regression_lock():
    """Every configured symbol resolves to its pre-refactor full config."""
    assert set(_SYMBOL_ML_CONFIG) == set(_EXPECTED_RESOLVED)
    for sym, expected in _EXPECTED_RESOLVED.items():
        assert _ml_config(sym) == expected


def test_override_wins_over_default():
    """A per-symbol override must beat the default for differing fields."""
    assert _ml_config("TLT")["model"] == "lgb"
    assert _DEFAULT_ML_CONFIG["model"] == "gb"
    assert _ml_config("SPY")["train_size"] == 2520
    assert _DEFAULT_ML_CONFIG["train_size"] == 1260
    assert _ml_config("QQQ")["feature_set"] == "vol_rich_plus"
    assert _DEFAULT_ML_CONFIG["feature_set"] == "vol_macro"


def test_default_fields_inherited_when_not_overridden():
    """Fields absent from a symbol's overrides come from the default unchanged."""
    qqq = _ml_config("QQQ")
    assert qqq["model"] == "gb"
    assert qqq["train_size"] == 1260
    assert qqq["target_horizon"] == 20
    assert qqq["use_all_train"] is True
    assert len(qqq) == len(_DEFAULT_ML_CONFIG)


def test_new_symbol_without_overrides_yields_default_config():
    """A symbol with no overrides must resolve exactly to the default config."""
    assert _ml_config("MISSING-SYMBOL") == _DEFAULT_ML_CONFIG
