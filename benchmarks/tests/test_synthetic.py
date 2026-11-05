"""Tests for benchmarks.review.synthetic: exact operating-point maths and the scenarios."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pytest

from benchmarks.review import synthetic

FPR = 0.02


def test_linear_tpr_matches_monte_carlo() -> None:
    g = synthetic.Biomarkers(d=5, delta=1.0, kappa=4.0)
    # 1M samples per class: the Monte Carlo standard error is about 0.0008, so 0.004 is ~5 SE.
    rng = np.random.default_rng(0)
    x, y = g.sample(2_000_000, 0.5, rng)
    for w in (np.eye(5)[0], np.r_[1.0, 0.7, 0.0, -0.2, 0.1]):
        s = x @ w
        threshold = np.quantile(s[y == 0], 1 - FPR)
        empirical = float((s[y == 1] > threshold).mean())
        assert synthetic.linear_tpr(w, g, FPR) == pytest.approx(empirical, abs=0.004)


def test_well_specified_case_has_no_headroom() -> None:
    # S-1: equal spread, so the best linear score is the mean-difference direction.
    g = synthetic.Biomarkers(d=5, delta=1.0, kappa=1.0)
    best, _ = synthetic.best_linear_tpr(g, FPR)
    z = NormalDist().inv_cdf(1 - FPR)
    assert best == pytest.approx(NormalDist().cdf(1.0 - z), abs=1e-4)
    assert synthetic.linear_tpr(np.eye(5)[0], g, FPR) == pytest.approx(best, abs=1e-4)


@pytest.mark.parametrize("kappa", [4.0, 8.0])
def test_more_variable_disease_creates_headroom_at_high_specificity(kappa: float) -> None:
    # S-2: the mean-difference direction is no longer the best at 98% specificity.
    g = synthetic.Biomarkers(d=5, delta=1.0, kappa=kappa)
    best, w = synthetic.best_linear_tpr(g, FPR)
    assert best - synthetic.linear_tpr(np.eye(5)[0], g, FPR) > 0.02
    assert abs(w[1]) > 0.1  # the best direction uses the second biomarker


def test_bayes_tpr_bounds_every_linear_score() -> None:
    g = synthetic.Biomarkers(d=5, delta=1.0, kappa=4.0)
    best, _ = synthetic.best_linear_tpr(g, FPR)
    bayes = synthetic.bayes_tpr(g, FPR, n=400_000, rng=np.random.default_rng(1))
    assert bayes >= best - 0.005


@pytest.mark.parametrize(
    "scenario",
    [
        synthetic.Biomarkers(d=6, delta=1.0, kappa=4.0),
        synthetic.LookAlikes(d=6, share=0.02),
        synthetic.RareSubtype(d=6, share=0.1),
    ],
)
def test_samplers_are_exact_and_reproducible(scenario: synthetic.Scenario) -> None:
    x, y = scenario.sample(10_000, 0.05, np.random.default_rng(2))
    x2, y2 = scenario.sample(10_000, 0.05, np.random.default_rng(2))
    assert x.shape == (10_000, 6)
    assert int(y.sum()) == 500
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)


def test_look_alikes_are_negatives_that_sit_with_the_positives() -> None:
    s = synthetic.LookAlikes(d=6, share=0.05)
    x, y, group = s.sample_with_groups(20_000, 0.1, np.random.default_rng(3))
    look = group == 1
    assert int(look.sum()) == round(0.05 * 18_000)
    assert np.all(y[look] == 0)
    # along the positives' direction the look-alikes rank with the positives, not the negatives
    assert x[look, 0].mean() == pytest.approx(x[y == 1, 0].mean(), abs=0.15)


def test_rare_subtype_share_and_labels() -> None:
    s = synthetic.RareSubtype(d=6, share=0.1)
    _, y, group = s.sample_with_groups(20_000, 0.1, np.random.default_rng(4))
    assert int((group == 1).sum()) == 200  # 10% of the 2,000 positives
    assert np.all(y[group >= 0] == 1)
    assert np.all(group[y == 0] == -1)


def test_make_task_builds_splits_and_an_oracle_sample() -> None:
    task = synthetic.make_task(
        "s1", synthetic.Biomarkers(d=5), prevalence=0.05, n_train=2000, n_test=5000, seed=0
    )
    assert len(task.y_train) == 2000
    assert len(task.y_val) == 2000
    assert len(task.y_test) == 5000
    assert task.oracle is not None
    assert len(task.oracle[1]) == 5000
    again = synthetic.make_task(
        "s1", synthetic.Biomarkers(d=5), prevalence=0.05, n_train=2000, n_test=5000, seed=0
    )
    assert np.array_equal(task.x_train.numpy(), again.x_train.numpy())
