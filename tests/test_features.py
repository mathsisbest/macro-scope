import numpy as np
import pandas as pd

from mmi.ml.features import feature_columns, make_features
from mmi.ml.forecast import SEED, make_regressor


def test_make_features_columns_and_no_leakage():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30),
            "close": np.linspace(100, 130, 30),
            "daily_return": np.linspace(0.01, 0.02, 30),
        }
    )
    feats = make_features(df)
    assert set(feature_columns()).issubset(feats.columns)
    # Target is the *next* return, so the final row's target must be NaN (no future leak).
    assert pd.isna(feats["target_next_ret"].iloc[-1])


def test_make_regressor_carries_variance_control_params():
    """Verify the factory sets max_depth and min_samples_leaf for variance control."""
    est = make_regressor(n_estimators=10)
    assert est.max_depth == 5, "max_depth must be 5 to limit tree depth at ~1e-4 daily-return scale"
    assert est.min_samples_leaf == 20, "min_samples_leaf must be 20 to prevent leaf overfitting"
    assert est.max_features == "sqrt"
    assert est.random_state == SEED


def test_make_regressor_seed_override():
    """Custom seed is forwarded correctly."""
    est = make_regressor(n_estimators=10, seed=42)
    assert est.random_state == 42
