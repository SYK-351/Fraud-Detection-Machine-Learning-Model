import numpy as np
import pandas as pd
import pytest

from fraud.data import chronological_split


def sample_frame(n=100):
    return pd.DataFrame({"Time": np.arange(n), "Class": np.tile([0, 0, 1, 0, 0], n // 5)})


def test_chronological_split_has_no_index_overlap_and_keeps_order():
    frame = sample_frame()
    split = chronological_split(frame)
    assert set(split.train.index).isdisjoint(split.validation.index)
    assert set(split.train.index).isdisjoint(split.test.index)
    assert set(split.validation.index).isdisjoint(split.test.index)
    assert split.train.Time.max() <= split.validation.Time.min()
    assert split.validation.Time.max() <= split.test.Time.min()


def test_chronological_split_covers_each_input_row_once():
    frame = sample_frame()
    split = chronological_split(frame.sample(frac=1, random_state=7))
    combined = pd.concat(tuple(split))
    assert len(combined) == len(frame)
    assert combined.index.nunique() == len(frame)


def test_invalid_split_fractions_raise():
    with pytest.raises(ValueError):
        chronological_split(sample_frame(), train_fraction=0.8, validation_fraction=0.3)


def test_exact_duplicates_do_not_cross_split_boundaries():
    frame = sample_frame()
    frame = pd.concat([frame, frame.iloc[[59]]], ignore_index=True)
    split = chronological_split(frame)
    combined = pd.concat(tuple(split))
    assert not combined.duplicated().any()


def test_legacy_split_is_stratified_and_uses_test_field():
    from fraud.data import legacy_split
    frame = sample_frame(500)
    split = legacy_split(frame, seed=9)
    assert len(split.validation) == 0
    assert len(split.test) == 100
    assert split.train.Class.mean() == pytest.approx(split.test.Class.mean(), abs=0.02)
