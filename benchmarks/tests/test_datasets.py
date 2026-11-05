"""Tests for benchmarks.review.datasets: published sizes, clean features, disjoint splits.

Each test needs its dataset in the local cache (``benchmarks.data.CACHE``) and is skipped,
with the reason, when it is absent: CI has no network access to the data sources.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.review import datasets, runner

# name -> (total samples, total positives) as published
SIZES = {
    "creditcard": (284_807, 492),
    "aps": (76_000, 1_375),
    "secom": (1_567, 104),
    "htru2": (17_898, 1_639),
    "mammography": (11_183, 260),
}


def load(name: str) -> runner.Task:
    if not datasets.available(name):
        pytest.skip(f"{name} is not in the local cache (run datasets.fetch({name!r}))")
    return datasets.load(name)


@pytest.mark.parametrize("name", sorted(SIZES))
def test_sizes_match_the_published_counts(name: str) -> None:
    task = load(name)
    ys = (task.y_train, task.y_val, task.y_test)
    total, positives = SIZES[name]
    assert sum(len(y) for y in ys) == total
    assert sum(int((y > 0.5).sum()) for y in ys) == positives


@pytest.mark.parametrize("name", sorted(SIZES))
def test_features_are_finite_and_standardised_on_train(name: str) -> None:
    task = load(name)
    for x in (task.x_train, task.x_val, task.x_test):
        assert x.dtype == torch.float32
        assert torch.isfinite(x).all()
    assert task.x_train.mean(0).abs().max() < 1e-3


@pytest.mark.parametrize("name", ["creditcard", "secom", "htru2"])
def test_random_splits_are_stratified_and_seeded(name: str) -> None:
    a, b = load(name), datasets.load(name, split_seed=1)
    rate = float((a.y_train > 0.5).float().mean())
    for y in (a.y_val, a.y_test):
        assert float((y > 0.5).float().mean()) == pytest.approx(rate, rel=0.15)
    assert not torch.equal(a.x_train[:20], b.x_train[:20])
    assert torch.equal(a.x_train, datasets.load(name).x_train)


def test_aps_keeps_the_official_test_set() -> None:
    task = load("aps")
    assert len(task.y_test) == 16_000
    assert int((task.y_test > 0.5).sum()) == 375


def test_every_source_names_a_licence() -> None:
    for source in datasets.SOURCES.values():
        assert source.licence
        assert source.citation
