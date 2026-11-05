"""Tests for benchmarks.review.manipulate (sub-situations U1, U2, U4, U5, U6)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.review import manipulate


def labels(n_pos: int, n_neg: int) -> np.ndarray:
    return np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(np.float32)


def test_subsample_fixed_positives_and_ratio() -> None:
    y = labels(300, 5000)
    idx = manipulate.subsample(y, np.random.default_rng(0), n_pos=100, ratio=20)
    assert int((y[idx] > 0.5).sum()) == 100
    assert int((y[idx] < 0.5).sum()) == 2000
    assert np.all(np.diff(idx) > 0)  # sorted, no duplicates


def test_subsample_defaults_keep_everything() -> None:
    y = labels(30, 70)
    np.testing.assert_array_equal(manipulate.subsample(y, np.random.default_rng(0)), np.arange(100))


def test_subsample_is_reproducible_and_seed_dependent() -> None:
    y = labels(300, 5000)
    a = manipulate.subsample(y, np.random.default_rng(1), n_pos=50, ratio=10)
    b = manipulate.subsample(y, np.random.default_rng(1), n_pos=50, ratio=10)
    c = manipulate.subsample(y, np.random.default_rng(2), n_pos=50, ratio=10)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(("n_pos", "ratio"), [(400, None), (100, 60.0)])
def test_subsample_refuses_impossible_designs(n_pos: int, ratio: float | None) -> None:
    with pytest.raises(ValueError, match="only"):
        manipulate.subsample(labels(300, 5000), np.random.default_rng(0), n_pos=n_pos, ratio=ratio)


def test_contamination_sets_the_share_of_wrong_positives() -> None:
    y = labels(900, 9100)
    noisy, flipped = manipulate.noisy_labels(y, np.random.default_rng(0), pos_contamination=0.1)
    label_one = noisy > 0.5
    assert flipped.sum() == 100  # 100 / (900 + 100) = 10% of label 1 is wrong
    assert np.all(y[flipped] == 0)
    assert np.all(noisy[flipped] == 1)
    assert flipped[label_one].mean() == pytest.approx(0.1)


def test_positive_flips_move_true_positives_to_label_zero() -> None:
    y = labels(200, 800)
    noisy, flipped = manipulate.noisy_labels(y, np.random.default_rng(0), pos_flip=0.25)
    assert flipped.sum() == 50
    assert np.all(y[flipped] == 1)
    assert np.all(noisy[flipped] == 0)


def test_no_noise_is_identity() -> None:
    y = labels(10, 90)
    noisy, flipped = manipulate.noisy_labels(y, np.random.default_rng(0))
    np.testing.assert_array_equal(noisy, y)
    assert not flipped.any()


def test_mix_subtypes_hits_the_requested_counts() -> None:
    subtype = np.r_[np.zeros(500), np.ones(300), np.full(200, 2), np.full(4000, -1)].astype(int)
    idx = manipulate.mix_subtypes(
        subtype, {0: 0.9, 1: 0.1}, n_pos=200, rng=np.random.default_rng(0)
    )
    counts = np.bincount(subtype[idx], minlength=3)
    np.testing.assert_array_equal(counts, [180, 20, 0])


def test_mix_subtypes_refuses_too_few() -> None:
    subtype = np.r_[np.zeros(10), np.ones(5)].astype(int)
    with pytest.raises(ValueError, match="only"):
        manipulate.mix_subtypes(subtype, {0: 0.5, 1: 0.5}, n_pos=20, rng=np.random.default_rng(0))


def test_minority_types() -> None:
    rng = np.random.default_rng(0)
    negatives = rng.normal(0.0, 1.0, size=(400, 2))
    cluster = rng.normal(10.0, 0.1, size=(20, 2))  # safe: neighbours are positives
    lonely = np.array([[0.0, 0.0]])  # outlier: inside the negatives
    x = np.vstack([cluster, lonely, negatives])
    y = np.r_[np.ones(21), np.zeros(400)].astype(np.float32)
    kinds = manipulate.minority_types(x, y)
    assert kinds.shape == (21,)
    assert set(kinds[:20]) == {"safe"}
    assert kinds[20] == "outlier"
