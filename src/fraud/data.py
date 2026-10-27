"""Dataset loading, feature construction, and leakage-safe splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import DATA_PATH, SEED, TEST_FRACTION, TRAIN_FRACTION, VALIDATION_FRACTION

from .features import FEATURE_COLUMNS, add_features

EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        yield self.train
        yield self.validation
        yield self.test


def load_transactions(path: str | Path = DATA_PATH, *, deduplicate: bool = True) -> pd.DataFrame:
    """Load and validate the ULB transactions, optionally dropping exact duplicates."""
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected columns {EXPECTED_COLUMNS}, got {list(frame.columns)}")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not set(frame["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must be binary with labels 0 and 1")
    if (frame["Amount"] < 0).any():
        raise ValueError("Amount must be nonnegative for log1p transform")
    if deduplicate:
        frame = frame.drop_duplicates().copy()
    return add_features(frame)


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> DataSplits:
    """Sort by transaction time, then allocate contiguous train/validation/test rows."""
    test_fraction = 1.0 - train_fraction - validation_fraction
    if train_fraction <= 0 or validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Split fractions must all be positive")
    ordered = frame.drop_duplicates().sort_values("Time", kind="mergesort")
    n = len(ordered)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    return DataSplits(
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def legacy_split(frame: pd.DataFrame, seed: int = SEED) -> DataSplits:
    """Reproducible random stratified 80/20 holdout used only by the ablation."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        frame, test_size=0.20, random_state=seed, stratify=frame["Class"]
    )
    return DataSplits(train.copy(), train.iloc[0:0].copy(), test.copy())


def audit_splits(frame: pd.DataFrame, splits: DataSplits) -> dict:
    """Return compact dataset provenance and per-split counts for JSON output."""
    def summary(part: pd.DataFrame) -> dict:
        return {
            "rows": int(len(part)),
            "frauds": int(part["Class"].sum()),
            "legitimate": int((part["Class"] == 0).sum()),
            "time_min_seconds": float(part["Time"].min()) if len(part) else None,
            "time_max_seconds": float(part["Time"].max()) if len(part) else None,
        }

    return {
        "rows_after_deduplication": int(len(frame)),
        "frauds_after_deduplication": int(frame["Class"].sum()),
        "time_span_hours": float((frame["Time"].max() - frame["Time"].min()) / 3600),
        "splits": {name: summary(part) for name, part in zip(("train", "validation", "test"), splits)},
    }
