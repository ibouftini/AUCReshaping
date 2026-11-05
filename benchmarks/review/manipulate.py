"""Controlled sub-situations on real or synthetic training data.

Every function acts on the training split only; validation and test sets stay untouched so
that results across levels of a manipulation are measured on the same data.
"""

from __future__ import annotations

import numpy as np


def subsample(
    y: np.ndarray,
    rng: np.random.Generator,
    n_pos: int | None = None,
    ratio: float | None = None,
) -> np.ndarray:
    """Sorted indices keeping ``n_pos`` positives and ``ratio`` negatives per positive.

    U1 (relative rarity) varies ``ratio`` at a fixed ``n_pos``; U2 (absolute rarity) varies
    ``n_pos`` at a fixed ``ratio``. ``None`` keeps all positives or all negatives. A design
    that asks for more samples than exist raises instead of silently changing the design.
    """
    pos, neg = np.flatnonzero(y > 0.5), np.flatnonzero(y <= 0.5)
    k_pos = len(pos) if n_pos is None else n_pos
    k_neg = len(neg) if ratio is None else round(ratio * k_pos)
    if k_pos > len(pos) or k_neg > len(neg):
        raise ValueError(
            f"asked for {k_pos} positives and {k_neg} negatives; there are only {len(pos)} "
            f"and {len(neg)}"
        )
    keep = np.concatenate(
        [rng.choice(pos, k_pos, replace=False), rng.choice(neg, k_neg, replace=False)]
    )
    return np.sort(keep)


def noisy_labels(
    y: np.ndarray,
    rng: np.random.Generator,
    pos_contamination: float = 0.0,
    pos_flip: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip labels; return the noisy labels and the boolean mask of flipped samples.

    ``pos_contamination`` is the share of the resulting label-1 class that are true
    negatives: they score low, so AUCReshaping boosts them (the risk in H3). ``pos_flip`` is
    the fraction of true positives relabelled 0: they join the reference class and push its
    upper tail, hence the threshold, up. Counts are exact (rounded), not Bernoulli draws.
    """
    pos, neg = np.flatnonzero(y > 0.5), np.flatnonzero(y <= 0.5)
    n_flip_pos = round(pos_flip * len(pos))
    kept_pos = len(pos) - n_flip_pos
    n_contam = round(pos_contamination * kept_pos / (1 - pos_contamination))
    if n_contam > len(neg):
        raise ValueError(f"needs {n_contam} negatives; there are only {len(neg)}")
    flipped = np.zeros(len(y), dtype=bool)
    flipped[rng.choice(pos, n_flip_pos, replace=False)] = True
    flipped[rng.choice(neg, n_contam, replace=False)] = True
    noisy = np.where(flipped, 1 - (y > 0.5), y > 0.5).astype(y.dtype)
    return noisy, flipped


def mix_subtypes(
    subtype: np.ndarray, proportions: dict[int, float], n_pos: int, rng: np.random.Generator
) -> np.ndarray:
    """Sorted indices of ``n_pos`` positives drawn with the given subtype proportions (U4).

    ``subtype`` holds a non-negative subtype id for each positive and -1 for negatives.
    Combine the result with negatives from :func:`subsample`.
    """
    chosen = []
    for kind, share in proportions.items():
        pool = np.flatnonzero(subtype == kind)
        k = round(share * n_pos)
        if k > len(pool):
            raise ValueError(f"subtype {kind}: asked for {k}; there are only {len(pool)}")
        chosen.append(rng.choice(pool, k, replace=False))
    return np.sort(np.concatenate(chosen))


def minority_types(x: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    """Kind of each positive from its k = 5 nearest neighbours (Napierala & Stefanowski 2016).

    Same-class neighbours 4-5: "safe", 2-3: "borderline", 1: "rare", 0: "outlier". Features
    are standardised first. Returns one string per positive, in the order of ``x[y > 0.5]``.
    """
    from sklearn.neighbors import NearestNeighbors

    z = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    positive = y > 0.5
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z[positive])
    same = positive[idx[:, 1:]].sum(axis=1)  # column 0 is the point itself
    kinds = np.array(["outlier", "rare", "borderline", "borderline", "safe", "safe"])
    return kinds[same]
