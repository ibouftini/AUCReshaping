"""From the independent review: how strong is a boost, measured on real scores?"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.review import boost_strength as bs


def _scores(seed: int = 0, n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.8).astype(np.float32)
    return (rng.normal(size=n) + 1.5 * y).astype(np.float32), y


def test_zero_boost_is_bce() -> None:
    z, y = _scores()
    out = bs.measure(z, y, [0.98], [0.0], combine="band", min_ratio=0.1)
    assert out["mass_ratio"] == pytest.approx(1.0)
    assert out["share_undefined"] == 0.0
    assert out["capped_share_of_extra"] == 0.0


def test_a_boost_above_every_probability_gives_exactly_the_cap() -> None:
    # Every positive is below the threshold and has p <= b, so each gets 1 / min_ratio times BCE.
    z = np.array([-3.0, -2.0, -2.5, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    y = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    out = bs.measure(z, y, [0.5], [0.99], combine="band", min_ratio=0.1)
    assert out["share_flagged"] == pytest.approx(1.0)
    assert out["share_undefined"] == pytest.approx(1.0)
    assert out["mass_ratio"] == pytest.approx(10.0, rel=1e-4)
    assert out["capped_share_of_extra"] == pytest.approx(1.0)


def test_a_larger_cap_ratio_weakens_the_boost() -> None:
    z, y = _scores()
    strong = bs.measure(z, y, [0.9, 0.98], [0.4, 0.1], combine="band", min_ratio=0.1)
    weak = bs.measure(z, y, [0.9, 0.98], [0.4, 0.1], combine="band", min_ratio=0.5)
    assert 1.0 < weak["mass_ratio"] < strong["mass_ratio"]


def test_sum_is_at_least_as_strong_as_band() -> None:
    z, y = _scores()
    band = bs.measure(z, y, [0.9, 0.98], [0.2, 0.2], combine="band", min_ratio=0.1)
    total = bs.measure(z, y, [0.9, 0.98], [0.2, 0.2], combine="sum", min_ratio=0.1)
    assert total["mass_ratio"] > band["mass_ratio"]


def test_constant_boost_under_band_equals_one_top_threshold() -> None:
    z, y = _scores()
    several = bs.measure(z, y, [0.9, 0.95, 0.98], [0.2, 0.2, 0.2], combine="band", min_ratio=0.1)
    one = bs.measure(z, y, [0.98], [0.2], combine="band", min_ratio=0.1)
    assert several["mass_ratio"] == pytest.approx(one["mass_ratio"])


def test_random_recipients_keep_the_count_but_can_change_the_strength() -> None:
    z, y = _scores()
    out = bs.measure(z, y, [0.98], [0.3], combine="band", min_ratio=0.1)
    assert 0.0 < out["random_extra_share"] < 1.0
    # Random recipients overlap the flagged set at about the flagged share.
    assert out["random_overlap"] == pytest.approx(out["share_flagged"], abs=0.05)


def test_the_threshold_comes_from_label_zero() -> None:
    z, y = _scores()
    flipped = bs.measure(-z, 1 - y, [0.98], [0.3], combine="band", min_ratio=0.1)
    plain = bs.measure(z, y, [0.98], [0.3], combine="band", min_ratio=0.1)
    assert flipped["share_flagged"] != pytest.approx(plain["share_flagged"], abs=1e-3)
