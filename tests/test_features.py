import numpy as np
import pandas as pd

from mmi.ml.features import feature_columns, make_features


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
