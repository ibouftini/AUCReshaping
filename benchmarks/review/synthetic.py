"""Synthetic scenarios. Each mimics a real situation and tests one effect.

- :class:`Biomarkers` (S-1, S-2): healthy cases are N(0, I); disease shifts biomarker 1 by
  ``delta`` and, for ``kappa`` > 1, makes biomarkers 1 and 2 vary together more
  (covariance I + (kappa - 1) u u^T with u = (e1 + e2) / sqrt(2)), as heterogeneous patients or
  irregular fraud amounts do. ``kappa`` = 1 is the case where logistic regression is the right
  model (S-1, the null control). The extra spread is one-sided on purpose: a symmetric one
  would make the best linear scores a mirror pair that no gradient method can choose between.
- :class:`LookAlikes` (S-3): a share of the negatives sits with the positives along biomarker
  1 and differs only along biomarker 2, like benign mimics of disease or legitimate purchases
  that look like fraud. They decide the high-specificity threshold.
- :class:`RareSubtype` (S-4): a share of the positives is a subtype that shows up along
  another biomarker, like an atypical presentation or a rare attack family.

For a linear score w, both classes' scores are Gaussian under :class:`Biomarkers`, so the
TPR at a fixed FPR has a closed form (:func:`linear_tpr`); :func:`best_linear_tpr` finds the
best linear score at an operating point, hence the exact headroom of a linear model, and
:func:`bayes_tpr` the Bayes optimum from the true likelihood ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

import numpy as np
import torch

from benchmarks.review.runner import Task


class Scenario(Protocol):
    def sample(
        self, n: int, prevalence: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def log_lr(self, x: np.ndarray) -> np.ndarray: ...


def _counts(n: int, prevalence: float) -> tuple[int, int]:
    n_pos = round(n * prevalence)
    return n - n_pos, n_pos


def _gaussian_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, (x - mean).T)
    return -0.5 * (z**2).sum(0) - np.log(np.diag(chol)).sum()


def _shuffled(
    rng: np.random.Generator, x: np.ndarray, y: np.ndarray, group: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = rng.permutation(len(y))
    return x[order].astype(np.float32), y[order].astype(np.float32), group[order]


@dataclass(frozen=True)
class Biomarkers:
    d: int = 20
    delta: float = 1.0
    kappa: float = 1.0

    def params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu0, cov0 = np.zeros(self.d), np.eye(self.d)
        u = np.zeros(self.d)
        u[:2] = 1 / np.sqrt(2)
        mu1 = np.zeros(self.d)
        mu1[0] = self.delta
        return mu0, cov0, mu1, cov0 + (self.kappa - 1) * np.outer(u, u)

    def sample(
        self, n: int, prevalence: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n0, n1 = _counts(n, prevalence)
        mu0, _, mu1, cov1 = self.params()
        neg = rng.standard_normal((n0, self.d)) + mu0
        pos = rng.standard_normal((n1, self.d)) @ np.linalg.cholesky(cov1).T + mu1
        x, y, _ = _shuffled(
            rng,
            np.vstack([neg, pos]),
            np.r_[np.zeros(n0), np.ones(n1)],
            np.r_[np.zeros(n0), np.ones(n1)],
        )
        return x, y

    def log_lr(self, x: np.ndarray) -> np.ndarray:
        mu0, cov0, mu1, cov1 = self.params()
        return _gaussian_logpdf(x, mu1, cov1) - _gaussian_logpdf(x, mu0, cov0)


def _iso_logpdf(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return -0.5 * ((x - mean) ** 2).sum(1)


def _mixture_logpdf(x: np.ndarray, means: list[np.ndarray], weights: list[float]) -> np.ndarray:
    parts = np.stack([np.log(w) + _iso_logpdf(x, m) for m, w in zip(means, weights, strict=True)])
    top = parts.max(0)
    return top + np.log(np.exp(parts - top).sum(0))


@dataclass(frozen=True)
class LookAlikes:
    d: int = 20
    share: float = 0.02  # of the negatives
    delta: float = 2.0  # positives' (and look-alikes') shift along biomarker 1
    gap: float = 2.0  # look-alikes' shift along biomarker 2

    def _means(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, pos, look = np.zeros(self.d), np.zeros(self.d), np.zeros(self.d)
        pos[0] = look[0] = self.delta
        look[1] = self.gap
        return base, look, pos

    def sample_with_groups(
        self, n: int, prevalence: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Group: 0 ordinary negative, 1 look-alike negative, -1 positive."""
        n0, n1 = _counts(n, prevalence)
        n_look = round(self.share * n0)
        base, look, pos = self._means()
        x = rng.standard_normal((n, self.d))
        x[: n0 - n_look] += base
        x[n0 - n_look : n0] += look
        x[n0:] += pos
        y = np.r_[np.zeros(n0), np.ones(n1)]
        group = np.r_[np.zeros(n0 - n_look), np.ones(n_look), -np.ones(n1)].astype(int)
        return _shuffled(rng, x, y, group)

    def sample(
        self, n: int, prevalence: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y, _ = self.sample_with_groups(n, prevalence, rng)
        return x, y

    def log_lr(self, x: np.ndarray) -> np.ndarray:
        base, look, pos = self._means()
        negative = _mixture_logpdf(x, [base, look], [1 - self.share, self.share])
        return _iso_logpdf(x, pos) - negative


@dataclass(frozen=True)
class RareSubtype:
    d: int = 20
    share: float = 0.1  # of the positives
    delta: float = 2.5  # common subtype, along biomarker 1
    rare_delta: float = 2.5  # rare subtype, along biomarker 3

    def _means(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, common, rare = np.zeros(self.d), np.zeros(self.d), np.zeros(self.d)
        common[0] = self.delta
        rare[2] = self.rare_delta
        return base, common, rare

    def sample_with_groups(
        self, n: int, prevalence: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Group: -1 negative, 0 common subtype, 1 rare subtype."""
        n0, n1 = _counts(n, prevalence)
        n_rare = round(self.share * n1)
        base, common, rare = self._means()
        x = rng.standard_normal((n, self.d))
        x[:n0] += base
        x[n0 : n - n_rare] += common
        x[n - n_rare :] += rare
        y = np.r_[np.zeros(n0), np.ones(n1)]
        group = np.r_[-np.ones(n0), np.zeros(n1 - n_rare), np.ones(n_rare)].astype(int)
        return _shuffled(rng, x, y, group)

    def sample(
        self, n: int, prevalence: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y, _ = self.sample_with_groups(n, prevalence, rng)
        return x, y

    def log_lr(self, x: np.ndarray) -> np.ndarray:
        base, common, rare = self._means()
        positive = _mixture_logpdf(x, [common, rare], [1 - self.share, self.share])
        return positive - _iso_logpdf(x, base)


def linear_tpr(w: np.ndarray, g: Biomarkers, fpr: float) -> float:
    """Exact TPR of the score ``x @ w`` at the threshold where the FPR is ``fpr``."""
    mu0, cov0, mu1, cov1 = g.params()
    s0, s1 = float(np.sqrt(w @ cov0 @ w)), float(np.sqrt(w @ cov1 @ w))
    threshold = float(w @ mu0) + s0 * NormalDist().inv_cdf(1 - fpr)
    return 1 - NormalDist().cdf((threshold - float(w @ mu1)) / s1)


def best_linear_tpr(
    g: Biomarkers, fpr: float, restarts: int = 16, seed: int = 0
) -> tuple[float, np.ndarray]:
    """Highest TPR of any linear score at ``fpr``, and its unit direction."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    starts = [np.eye(g.d)[0], *rng.standard_normal((restarts, g.d))]
    best, best_w = -1.0, starts[0]
    for w0 in starts:
        result = minimize(lambda w: -linear_tpr(w / np.linalg.norm(w), g, fpr), w0, method="BFGS")
        w = result.x / np.linalg.norm(result.x)
        tpr = linear_tpr(w, g, fpr)
        if tpr > best:
            best, best_w = tpr, w
    return best, best_w


def bayes_tpr(scenario: Scenario, fpr: float, n: int, rng: np.random.Generator) -> float:
    """TPR at ``fpr`` of the true likelihood ratio, by Monte Carlo on n samples per class."""
    x, y = scenario.sample(2 * n, 0.5, rng)
    llr = scenario.log_lr(x.astype(np.float64))
    threshold = np.quantile(llr[y == 0], 1 - fpr)
    return float((llr[y == 1] > threshold).mean())


def make_task(
    name: str,
    scenario: Scenario,
    prevalence: float,
    n_train: int,
    n_test: int,
    seed: int,
    n_val: int | None = None,
    n_oracle: int | None = None,
    test_scenario: Scenario | None = None,
) -> Task:
    """Train, validation, test and oracle samples, drawn in that order from one seed.

    ``test_scenario`` replaces the scenario for the test set only (deployment drift, S-8);
    the oracle sample stays with the training distribution.
    """
    rng = np.random.default_rng(seed)
    sizes = (n_train, n_val or n_train, n_test, n_oracle or n_test)
    sources = (scenario, scenario, test_scenario or scenario, scenario)
    parts, groups = [], []
    for size, scenario in zip(sizes, sources, strict=True):
        if hasattr(scenario, "sample_with_groups"):
            x, y, group = scenario.sample_with_groups(size, prevalence, rng)
        else:
            x, y = scenario.sample(size, prevalence, rng)
            group = np.where(y > 0.5, -1, 0)
        parts.append((torch.from_numpy(x), torch.from_numpy(y)))
        groups.append(group)
    (xt, yt), (xv, yv), (xs, ys), oracle = parts
    meta = {"scenario": repr(scenario), "test_groups": groups[2], "train_groups": groups[0]}
    return Task(name, xt, yt, xv, yv, xs, ys, oracle=oracle, meta=meta)
