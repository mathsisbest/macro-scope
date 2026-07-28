"""Time-series walk-forward and train/test splitters for Macro Scope ML forecasting."""

from collections.abc import Generator

import pandas as pd


def walk_forward_split(
    total_len: int,
    train_size: int,
    test_size: int,
    single_split: bool = False,
    use_all_train: bool = False,
    embargo: int = 0,
) -> Generator[tuple[list[int], list[int]], None, None]:
    """Generate (train_indices, test_indices) tuples for walk-forward or single-split backtests.

    Parameters
    ----------
    total_len : int
        Total length of the dataset.
    train_size : int
        Number of samples in each training window.
    test_size : int
        Number of samples in each test window.
    single_split : bool, default False
        If True, returns a single split with all remaining samples in the test set.
    use_all_train : bool, default False
        If True, expanding window training is used (train_start is always 0).
    embargo : int, default 0
        Number of samples to gap between end of training and test set to prevent leakage.

    Yields
    ------
    tuple[list[int], list[int]]
        (train_indices, test_indices)
    """
    if total_len < train_size + 1:
        return

    if single_split:
        train_idx = list(range(0, train_size))
        test_start = min(train_size + embargo, total_len)
        test_idx = list(range(test_start, total_len))
        if test_idx:
            yield train_idx, test_idx
        return

    step = test_size
    starts = range(0, total_len - train_size - embargo, step)
    for start in starts:
        train_start = 0 if use_all_train else start
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = min(test_start + test_size, total_len)
        if test_end <= test_start:
            break
        yield list(range(train_start, train_end)), list(range(test_start, test_end))


def feasible_date_range(df: pd.DataFrame, train_size: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (first_test_date, last_test_date) so that at least one full window exists.

    Returns (pd.NaT, pd.NaT) when no feasible range exists.
    """
    if len(df) < train_size + 1 or "date" not in df.columns:
        return pd.NaT, pd.NaT
    return df["date"].iloc[train_size], df["date"].iloc[-1]
