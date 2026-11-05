"""Tests for benchmarks.review.stats (run with ``uv run pytest benchmarks/tests``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.review import stats


def positive_mean(scores: np.ndarray, y: np.ndarray) -> float:
    return float(scores[y > 0.5].mean())


def make_group(rng: np.random.Generator, seeds: int = 5, n: int = 200) -> stats.PairedGroup:
    y = (rng.random(n) < 0.3).astype(np.float32)
    a = rng.normal(size=(seeds, n))
    return stats.PairedGroup(y=y, a=a, b=a.copy())


def test_identical_arms_give_zero_effect_and_p_one() -> None:
    group = make_group(np.random.default_rng(0))
    est = stats.two_level_bootstrap([group], positive_mean, n_boot=200)
    assert est.point == 0.0
    assert (est.lo, est.hi) == (0.0, 0.0)
    assert est.p_value == 1.0
    assert est.seeds_with_gain == 0
    assert est.cells == 5


def test_constant_shift_is_recovered_exactly() -> None:
    group = make_group(np.random.default_rng(1))
    group = stats.PairedGroup(y=group.y, a=group.a, b=group.a + 1.0)
    est = stats.two_level_bootstrap([group], positive_mean, n_boot=200)
    assert est.point == pytest.approx(1.0)
    assert est.lo == pytest.approx(1.0)
    assert est.hi == pytest.approx(1.0)
    assert est.p_value <= 2 / 201
    assert est.seeds_with_gain == 5


def test_seed_variance_sets_the_interval_width() -> None:
    # Each seed's difference is a constant offset, so only seed resampling adds variance.
    rng = np.random.default_rng(2)
    seeds, n = 10, 50
    y = np.r_[np.ones(10), np.zeros(n - 10)].astype(np.float32)
    offsets = rng.normal(0.5, 1.0, size=seeds)
    a = np.zeros((seeds, n))
    b = np.repeat(offsets[:, None], n, axis=1)
    est = stats.two_level_bootstrap([stats.PairedGroup(y, a, b)], positive_mean, n_boot=2000)
    assert est.point == pytest.approx(offsets.mean())
    assert est.lo < offsets.mean() < est.hi
    se = offsets.std(ddof=1) / math.sqrt(seeds)
    assert 2.5 * se < est.hi - est.lo < 4.5 * se  # a normal 95% interval spans 3.92 se


def test_test_cases_are_resampled_within_class() -> None:
    y = np.r_[np.ones(7), np.zeros(13)].astype(np.float32)
    idx = stats.stratified_indices(y, np.random.default_rng(3))
    assert len(idx) == 20
    assert int((y[idx] > 0.5).sum()) == 7


def test_several_groups_pool_all_cells() -> None:
    rng = np.random.default_rng(4)
    g1 = make_group(rng, seeds=3, n=100)
    g2 = make_group(rng, seeds=2, n=150)
    g1 = stats.PairedGroup(g1.y, g1.a, g1.a + 1.0)
    g2 = stats.PairedGroup(g2.y, g2.a, g2.a + 2.0)
    est = stats.two_level_bootstrap([g1, g2], positive_mean, n_boot=300)
    assert est.cells == 5
    assert est.point == pytest.approx((3 * 1.0 + 2 * 2.0) / 5)
    assert 1.0 - 1e-9 <= est.lo <= est.hi <= 2.0 + 1e-9


def test_shape_mismatch_is_rejected() -> None:
    y = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        stats.PairedGroup(y=y, a=np.zeros((2, 10)), b=np.zeros((3, 10)))


def test_random_effects_without_heterogeneity() -> None:
    pooled = stats.random_effects([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert pooled.q == pytest.approx(2.0)
    assert pooled.tau2 == 0.0
    assert pooled.i2 == 0.0
    assert pooled.mu == pytest.approx(2.0)
    assert pooled.se == pytest.approx(1 / math.sqrt(3))


def test_random_effects_with_heterogeneity() -> None:
    # Q = 8, df = 1, C = 1  ->  tau^2 = 7, weights 1/8 each, se = 2, I^2 = 7/8.
    pooled = stats.random_effects([0.0, 4.0], [1.0, 1.0])
    assert pooled.tau2 == pytest.approx(7.0)
    assert pooled.mu == pytest.approx(2.0)
    assert pooled.se == pytest.approx(2.0)
    assert pooled.i2 == pytest.approx(7 / 8)
    assert pooled.lo == pytest.approx(2.0 - 1.959964 * 2.0, abs=1e-4)


def test_random_effects_single_study() -> None:
    pooled = stats.random_effects([0.7], [0.2])
    assert (pooled.mu, pooled.se, pooled.tau2) == (pytest.approx(0.7), pytest.approx(0.2), 0.0)


def test_holm() -> None:
    adjusted = stats.holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    np.testing.assert_allclose(stats.holm([0.5, 0.9]), [1.0, 1.0])


def test_bayesian_signed_rank_clear_gain() -> None:
    p_left, p_rope, p_right = stats.bayesian_signed_rank(np.full(10, 5.0), rope=1.0)
    assert p_right > 0.95
    assert p_left + p_rope + p_right == pytest.approx(1.0)


def test_bayesian_signed_rank_equivalence() -> None:
    diffs = np.random.default_rng(5).uniform(-0.2, 0.2, size=20)
    _, p_rope, _ = stats.bayesian_signed_rank(diffs, rope=1.0)
    assert p_rope > 0.95


def test_bayesian_signed_rank_is_symmetric() -> None:
    diffs = np.random.default_rng(6).normal(0.0, 3.0, size=15)
    left, rope, right = stats.bayesian_signed_rank(diffs, rope=1.0, seed=0)
    right2, rope2, left2 = stats.bayesian_signed_rank(-diffs, rope=1.0, seed=0)
    assert (left, rope, right) == pytest.approx((left2, rope2, right2))


def test_min_detectable_effect() -> None:
    assert stats.min_detectable_effect(1.0, 1) == pytest.approx(2.8016, abs=1e-3)
    assert stats.min_detectable_effect(2.0, 16) == pytest.approx(2 * 2.8016 / 4, abs=1e-3)


def test_relative_change() -> None:
    assert stats.relative_change(0.802, 0.844) == pytest.approx(5.24, abs=0.01)
    assert math.isnan(stats.relative_change(0.0, 0.1))


def _clustered(
    rng: np.random.Generator, patients: int = 300, images: int = 4, seeds: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Images of one patient share a label and an effect, so they are far from independent."""
    ids = np.repeat(np.arange(patients), images)
    label = rng.integers(0, 2, patients)
    effect = rng.normal(0, 1.0, patients)
    y = label[ids].astype(float)
    a = np.stack([effect[ids] + rng.normal(0, 0.01, len(ids)) for _ in range(seeds)])
    b = a + effect[ids] * 0.0
    return y, ids, a, b


def _mean_score(scores: np.ndarray, y: np.ndarray) -> float:
    return float(scores.mean())


def test_patient_bootstrap_identical_arms_give_exactly_zero() -> None:
    y, ids, a, _ = _clustered(np.random.default_rng(0))
    est = stats.patient_bootstrap(y, ids, a, a.copy(), _mean_score, n_boot=50)
    assert est.point == 0.0
    assert est.lo == 0.0
    assert est.hi == 0.0
    assert est.p_value == 1.0


def test_patient_bootstrap_recovers_a_constant_shift() -> None:
    y, ids, a, _ = _clustered(np.random.default_rng(1))
    est = stats.patient_bootstrap(y, ids, a, a + 2.0, _mean_score, n_boot=50)
    assert est.point == pytest.approx(2.0)
    assert est.lo == pytest.approx(2.0)
    assert est.hi == pytest.approx(2.0)
    assert est.seeds_with_gain == len(a)


def test_patient_bootstrap_keeps_a_patients_images_together() -> None:
    # An effect shared by all of a patient's images: resampling images would understate the
    # interval by about the square root of the images per patient; resampling patients does not.
    rng = np.random.default_rng(2)
    patients, images = 200, 16
    ids = np.repeat(np.arange(patients), images)
    y = np.zeros(len(ids))
    effect = rng.normal(0, 1.0, patients)[ids]
    a = np.zeros((1, len(ids)))
    b = effect[None]
    clustered = stats.patient_bootstrap(y, ids, a, b, _mean_score, n_boot=400, resample="patients")
    independent = stats.patient_bootstrap(
        y, np.arange(len(ids)), a, b, _mean_score, n_boot=400, resample="patients"
    )
    ratio = (clustered.hi - clustered.lo) / (independent.hi - independent.lo)
    assert 2.5 < ratio < 5.5


def test_patient_bootstrap_can_isolate_each_source_of_uncertainty() -> None:
    rng = np.random.default_rng(3)
    y, ids, a, _ = _clustered(rng)
    # Differences vary by seed, and not at all by patient.
    b = a + rng.normal(0, 1.0, (len(a), 1))
    seeds_only = stats.patient_bootstrap(y, ids, a, b, _mean_score, n_boot=200, resample="seeds")
    patients_only = stats.patient_bootstrap(
        y, ids, a, b, _mean_score, n_boot=200, resample="patients"
    )
    assert seeds_only.hi - seeds_only.lo > 0.1
    assert patients_only.hi - patients_only.lo == pytest.approx(0.0)


def test_patient_bootstrap_refuses_an_unknown_resampling() -> None:
    y, ids, a, _ = _clustered(np.random.default_rng(4))
    with pytest.raises(ValueError, match="resample"):
        stats.patient_bootstrap(y, ids, a, a, _mean_score, n_boot=2, resample="images")


def test_patient_bootstrap_refuses_mismatched_patients() -> None:
    y, ids, a, _ = _clustered(np.random.default_rng(5))
    with pytest.raises(ValueError, match="patients"):
        stats.patient_bootstrap(y, ids[:-1], a, a, _mean_score, n_boot=2)


def test_patient_bootstrap_is_repeatable() -> None:
    rng = np.random.default_rng(6)
    y, ids, a, _ = _clustered(rng)
    b = a + rng.normal(0, 0.5, a.shape)
    one = stats.patient_bootstrap(y, ids, a, b, _mean_score, n_boot=20, seed=7)
    two = stats.patient_bootstrap(y, ids, a, b, _mean_score, n_boot=20, seed=7)
    assert one == two


def test_paired_t_interval_matches_the_textbook_values() -> None:
    lo, hi = stats.paired_t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    half = 2.776 * np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1) / np.sqrt(5)
    assert lo == pytest.approx(3.0 - half, abs=1e-3)
    assert hi == pytest.approx(3.0 + half, abs=1e-3)


def test_paired_t_interval_needs_two_seeds_and_sets_the_quantile_past_the_table() -> None:
    with pytest.raises(ValueError, match="two"):
        stats.paired_t_interval([1.0])
    lo, hi = stats.paired_t_interval([0.0, 1.0] * 20)
    half = 2.0 * np.std([0.0, 1.0] * 20, ddof=1) / np.sqrt(40)
    assert (hi - lo) / 2 == pytest.approx(half, abs=1e-9)


def test_the_conservative_interval_is_the_wider_on_each_side() -> None:
    assert stats.conservative_interval((-1.0, 2.0), (-0.5, 3.0)) == (-1.0, 3.0)


def test_the_paired_t_p_value_matches_the_quantile_table() -> None:
    # A t statistic at the 97.5% quantile gives a two-sided p of 0.05.
    for n, quantile in ((6, 2.571), (10, 2.262), (20, 2.093)):
        values = np.array([1.0, -1.0] * (n // 2))
        values = values / values.std(ddof=1) + quantile / math.sqrt(n)
        assert stats.paired_t_p_value(values.tolist()) == pytest.approx(0.05, abs=2e-3)


def test_the_paired_t_p_value_handles_no_spread() -> None:
    assert stats.paired_t_p_value([0.0, 0.0, 0.0]) == 1.0
    assert stats.paired_t_p_value([1.0, 1.0, 1.0]) == 0.0
    assert stats.paired_t_p_value([0.0, 0.0]) == 1.0
