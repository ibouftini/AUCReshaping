"""Evaluation and paired statistics for the benchmarks."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from aucreshape import metrics


def evaluate(
    val_logits: Tensor,
    val_y: Tensor,
    test_logits: Tensor,
    test_y: Tensor,
    specificities: Sequence[float],
    max_fpr: float = 0.05,
) -> dict[str, float]:
    """Test metrics, binary only. Thresholds for the ``*_valthr`` entries come from validation."""
    out: dict[str, float] = {
        "auc": float(metrics.partial_auc(test_logits, test_y)),
        f"pauc{max_fpr:g}": float(
            metrics.partial_auc(test_logits, test_y, max_fpr, standardized=True)
        ),
    }
    negatives = test_logits[test_y < 0.5]
    for s in specificities:
        thr = metrics.threshold_at_specificity(val_logits, val_y, s)
        out[f"sens@{s:g}"] = float(metrics.sensitivity_at_specificity(test_logits, test_y, s))
        out[f"sens@{s:g}_valthr"] = float(
            metrics.sensitivity_at_threshold(test_logits, test_y, thr)
        )
        out[f"spec@{s:g}_valthr"] = float((negatives < thr).float().mean())
    return out


def sens_fn(s: float) -> Callable[[np.ndarray, np.ndarray], float]:
    def fn(scores: np.ndarray, y: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(
                metrics.sensitivity_at_specificity(torch.from_numpy(scores), torch.from_numpy(y), s)
            )

    return fn


def auc_fn(scores: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, scores))


def paired_bootstrap(
    y: np.ndarray,
    scores_a: Sequence[np.ndarray],
    scores_b: Sequence[np.ndarray],
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean over seeds of fn(b) - fn(a), with a 95% CI from a stratified paired bootstrap.

    Each replicate resamples the test set (stratified by class, the same indices for both
    methods and all seeds) and averages the per-seed differences.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y > 0.5), np.flatnonzero(y < 0.5)
    point = float(np.mean([fn(b, y) - fn(a, y) for a, b in zip(scores_a, scores_b, strict=True)]))
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy = y[idx]
        reps.append(
            np.mean(
                [fn(b[idx], yy) - fn(a[idx], yy) for a, b in zip(scores_a, scores_b, strict=True)]
            )
        )
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def mean_std(values: Sequence[float]) -> str:
    arr = np.asarray(values, dtype=float)
    std = arr.std(ddof=1) if len(arr) > 1 else math.nan
    return f"{arr.mean():.4f} +/- {std:.4f}"
