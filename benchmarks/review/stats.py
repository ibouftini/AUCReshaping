"""Statistics of the evaluation.

- ``two_level_bootstrap``: paired delta = fn(b) - fn(a) with a CI that resamples the units of
  training randomness (groups, e.g. splits, then seeds) and the test cases (stratified by
  class, shared by both arms and all seeds of a group, since the test set is common to them).
- ``patient_bootstrap``: the same contrast for one test set whose images cluster within patients;
  it resamples seeds and patients, keeping each patient's images together.
- ``paired_t_interval``, ``conservative_interval``: the guard against undercoverage with few seeds.
- ``random_effects``: DerSimonian-Laird pooling across settings, with I^2.
- ``holm``, ``bayesian_signed_rank`` (Benavoli et al. 2017), ``min_detectable_effect``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class PairedGroup:
    """One test set ``y`` (n,) and the scores of arms ``a`` and ``b`` per seed, (S, n) each."""

    y: np.ndarray
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        if self.a.shape != self.b.shape or self.a.ndim != 2 or self.a.shape[1] != len(self.y):
            raise ValueError(
                f"scores must have shape (S, n) with n = len(y); got a {self.a.shape}, "
                f"b {self.b.shape}, y {self.y.shape}"
            )


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    p_value: float
    cells: int  # (group, seed) pairs
    seeds_with_gain: int


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap indices that keep the number of positives and negatives."""
    pos, neg = np.flatnonzero(y > 0.5), np.flatnonzero(y <= 0.5)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def two_level_bootstrap(
    groups: Sequence[PairedGroup],
    fn: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Estimate:
    """Mean over all (group, seed) cells of fn(b) - fn(a), with a percentile CI.

    Each replicate resamples groups, then seeds within each drawn group, then that group's
    test cases. The two-sided p-value is ``2 * min(P(delta* <= 0), P(delta* >= 0))`` with add-one
    smoothing; it is 1 when every replicate is exactly 0.
    """
    diffs = [np.array([fn(g.b[s], g.y) - fn(g.a[s], g.y) for s in range(len(g.a))]) for g in groups]
    cells = np.concatenate(diffs)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for r in range(n_boot):
        values = []
        for gi in rng.integers(len(groups), size=len(groups)):
            g = groups[gi]
            idx = stratified_indices(g.y, rng)
            yy = g.y[idx]
            values.extend(
                fn(g.b[s][idx], yy) - fn(g.a[s][idx], yy)
                for s in rng.integers(len(g.a), size=len(g.a))
            )
        reps[r] = np.mean(values)
    return _estimate(reps, cells, level)


def _estimate(reps: np.ndarray, cells: np.ndarray, level: float) -> Estimate:
    alpha = (1 - level) / 2
    lo, hi = np.quantile(reps, [alpha, 1 - alpha])
    if np.all(reps == 0):
        p = 1.0
    else:
        tail = min((reps <= 0).sum(), (reps >= 0).sum())
        p = min(1.0, 2 * (float(tail) + 1) / (len(reps) + 1))
    return Estimate(
        point=float(cells.mean()),
        lo=float(lo),
        hi=float(hi),
        p_value=p,
        cells=len(cells),
        seeds_with_gain=int((cells > 0).sum()),
    )


def patient_bootstrap(
    y: np.ndarray,
    patients: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    fn: Metric,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    resample: str = "both",
) -> Estimate:
    """Mean over seeds of fn(b) - fn(a), with a percentile CI that respects patients.

    Each replicate draws patients with replacement, keeping all of a patient's images, and seeds
    with replacement; both arms and all seeds see the same draw.

    Parameters
    ----------
    y, patients : ndarray, shape (n,)
        Labels and patient identifiers of the test images.
    a, b : ndarray, shape (S, n)
        Scores of the two arms, paired by seed.
    fn : callable
        Takes scores and labels, shape (m,) each, and returns the metric.
    n_boot : int, default 2000
        Number of replicates.
    seed : int, default 0
        Seed of the resampling.
    level : float, default 0.95
        Coverage of the interval.
    resample : {"both", "seeds", "patients"}, default "both"
        What to resample. One source alone shows how much of the interval comes from it.

    Returns
    -------
    Estimate
        The difference, its interval and two-sided p-value, and the seeds with a gain.
    """
    if resample not in ("both", "seeds", "patients"):
        raise ValueError(f"resample must be 'both', 'seeds' or 'patients'; got {resample!r}.")
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(y):
        raise ValueError(f"scores must have shape (S, n); got a {a.shape}, b {b.shape}.")
    if len(patients) != len(y):
        raise ValueError(f"patients must have one entry per image; got {len(patients)}, {len(y)}.")
    order = np.argsort(patients, kind="stable")
    _, starts = np.unique(patients[order], return_index=True)
    members = np.split(order, starts[1:])
    seeds = len(a)
    cells = np.array([fn(b[s], y) - fn(a[s], y) for s in range(seeds)])
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for r in range(n_boot):
        draw = rng.integers(seeds, size=seeds) if resample != "patients" else np.arange(seeds)
        if resample == "seeds":
            reps[r] = cells[draw].mean()
            continue
        chosen = rng.integers(len(members), size=len(members))
        idx = np.concatenate([members[i] for i in chosen])
        yy = y[idx]
        used, counts = np.unique(draw, return_counts=True)
        total = sum(
            count * (fn(b[s][idx], yy) - fn(a[s][idx], yy))
            for s, count in zip(used, counts, strict=True)
        )
        reps[r] = total / seeds
    return _estimate(reps, cells, level)


# Two-sided 97.5% quantiles of the t distribution, for 1 to 30 degrees of freedom.
T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def paired_t_interval(per_seed: Sequence[float]) -> tuple[float, float]:
    """Return the 95% t interval of per-seed differences; beyond 30 df the quantile is 2.0."""
    n = len(per_seed)
    if n < 2:
        raise ValueError("The paired t interval needs at least two seeds.")
    values = np.asarray(per_seed, dtype=float)
    quantile = T975[n - 2] if n - 2 < len(T975) else 2.0
    half = quantile * float(values.std(ddof=1)) / math.sqrt(n)
    return float(values.mean()) - half, float(values.mean()) + half


def paired_t_p_value(per_seed: Sequence[float]) -> float:
    """Return the two-sided p-value of a paired t test that the mean difference is zero."""
    n = len(per_seed)
    if n < 2:
        raise ValueError("The paired t test needs at least two seeds.")
    values = np.asarray(per_seed, dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    if sd == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    t = abs(mean) / (sd / math.sqrt(n))
    df = n - 1
    # With t = sqrt(df) tan(a), the density in a is proportional to cos(a)^(df - 1).
    grid = np.linspace(0.0, math.atan(t / math.sqrt(df)), 20_001)
    density = np.cos(grid) ** (df - 1)
    area = float(np.sum((density[1:] + density[:-1]) / 2 * np.diff(grid)))
    scale = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(math.pi)
    return min(1.0, max(0.0, 1.0 - 2.0 * scale * area))


def conservative_interval(
    one: tuple[float, float], other: tuple[float, float]
) -> tuple[float, float]:
    """Return the wider limit on each side of two intervals."""
    return min(one[0], other[0]), max(one[1], other[1])


@dataclass(frozen=True)
class Pooled:
    mu: float
    se: float
    lo: float
    hi: float
    tau2: float
    i2: float
    q: float


def random_effects(effects: Sequence[float], ses: Sequence[float], level: float = 0.95) -> Pooled:
    """DerSimonian-Laird random-effects pooling of per-setting effects and standard errors."""
    y, se = np.asarray(effects, dtype=float), np.asarray(ses, dtype=float)
    w = 1 / se**2
    fixed = float((w * y).sum() / w.sum())
    q = float((w * (y - fixed) ** 2).sum())
    df = len(y) - 1
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c) if df > 0 else 0.0
    w_star = 1 / (se**2 + tau2)
    mu = float((w_star * y).sum() / w_star.sum())
    se_mu = float(math.sqrt(1 / w_star.sum()))
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return Pooled(mu, se_mu, mu - z * se_mu, mu + z * se_mu, tau2, i2, q)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def bayesian_signed_rank(
    diffs: Sequence[float],
    rope: float,
    prior: float = 1.0,
    n_samples: int = 20000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Posterior probabilities that the difference is below, inside or above the ROPE.

    Bayesian signed-rank test (Benavoli et al. 2017): a Dirichlet process with a
    pseudo-observation at 0 of weight ``prior``. Each posterior sample gives the masses of
    pairwise means (z_i + z_j)/2 below -rope, within +/-rope and above +rope; the returned
    probabilities are how often each of the three is the largest.
    """
    z = np.concatenate([[0.0], np.asarray(diffs, dtype=float)])
    pair = z[:, None] + z[None, :]
    masks = [pair < -2 * rope, np.abs(pair) <= 2 * rope, pair > 2 * rope]
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.r_[prior, np.ones(len(z) - 1)], size=n_samples)
    theta = np.stack([((weights @ m) * weights).sum(axis=1) for m in masks], axis=1)
    winner = theta.argmax(axis=1)
    left, inside, right = (float((winner == k).mean()) for k in range(3))
    return left, inside, right


def min_detectable_effect(sd: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true mean difference a paired test on n seeds detects with the given power."""
    normal = NormalDist()
    return (normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)) * sd / math.sqrt(n)


def relative_change(a: float, b: float) -> float:
    """(b - a) / a in percent; NaN when a is 0."""
    return math.nan if a == 0 else 100 * (b - a) / a
