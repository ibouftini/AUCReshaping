"""ROC-region metrics: thresholds, sensitivity at specificity and partial AUC.

These functions read values back to the host and are meant for evaluation, not for the
training forward pass. Scores can be logits or probabilities (any monotone score); the
prediction rule is *positive iff score >= threshold*. Ties in scores form a single ROC
vertex, as in ``sklearn.metrics.roc_curve``. Targets of exactly 0.5 and NaN targets are
treated as missing, like masked elements.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from aucreshape._validation import percent_hint

__all__ = [
    "partial_auc",
    "sensitivity_at_specificity",
    "sensitivity_at_threshold",
    "threshold_at_specificity",
]

_TOL = 1e-9


class _Roc(NamedTuple):
    """Empirical ROC of one label, starting at (0, 0) with threshold +inf (float64, CPU)."""

    fpr: Tensor
    tpr: Tensor
    thresholds: Tensor
    num_pos: int
    num_neg: int


def _labels(
    scores: Tensor, targets: Tensor, mask: Tensor | None
) -> tuple[list[tuple[Tensor, Tensor]], bool]:
    """Split inputs into per-label ``(scores, is_positive)`` pairs on the CPU in float64."""
    if not isinstance(scores, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("scores and targets must be torch.Tensor objects.")
    if scores.dim() not in (1, 2):
        raise ValueError(f"scores must have shape (N,) or (N, C); got {tuple(scores.shape)}.")
    s = scores if scores.dim() == 2 else scores.unsqueeze(-1)
    if targets.numel() != s.numel() or (targets.dim() == 2 and targets.shape != s.shape):
        raise ValueError(
            f"targets must have the same shape as scores {tuple(scores.shape)}; "
            f"got {tuple(targets.shape)}."
        )
    y = targets.reshape(s.shape)
    if mask is not None:
        if mask.numel() != s.numel() or (mask.dim() == 2 and mask.shape != s.shape):
            raise ValueError(
                f"mask must have the same shape as scores {tuple(scores.shape)}; "
                f"got {tuple(mask.shape)}."
            )
        keep = mask.reshape(s.shape).detach().cpu() != 0
    else:
        keep = torch.ones(s.shape, dtype=torch.bool)
    s64 = s.detach().cpu().to(torch.float64)
    y_cpu = y.detach().cpu().to(torch.float64)
    pairs = []
    for c in range(s.shape[1]):
        k = keep[:, c] & (y_cpu[:, c] != 0.5) & ~torch.isnan(y_cpu[:, c])  # NaN = missing
        pairs.append((s64[k, c], y_cpu[k, c] > 0.5))
    return pairs, s.shape[1] == 1


def _roc(scores: Tensor, positive: Tensor) -> _Roc:
    order = torch.argsort(scores, descending=True, stable=True)
    s = scores[order]
    pos = positive[order].to(torch.float64)
    tps = torch.cumsum(pos, 0)
    fps = torch.cumsum(1.0 - pos, 0)
    if s.numel() > 0:
        last_of_tie = torch.cat(
            [torch.nonzero(s[1:] != s[:-1]).flatten(), torch.tensor([s.numel() - 1])]
        )
        tps, fps, thr = tps[last_of_tie], fps[last_of_tie], s[last_of_tie]
    else:
        thr = s
    zero = torch.zeros(1, dtype=torch.float64)
    tps = torch.cat([zero, tps])
    fps = torch.cat([zero, fps])
    thr = torch.cat([torch.tensor([math.inf], dtype=torch.float64), thr])
    num_pos = int(positive.sum())
    num_neg = int(positive.numel() - num_pos)
    fpr = fps / num_neg if num_neg > 0 else torch.full_like(fps, math.nan)
    tpr = tps / num_pos if num_pos > 0 else torch.full_like(tps, math.nan)
    return _Roc(fpr, tpr, thr, num_pos, num_neg)


def _specificities(specificity: float | Sequence[float]) -> tuple[list[float], bool]:
    if isinstance(specificity, (int, float)):
        values = [float(specificity)]
        scalar = True
    else:
        values = [float(s) for s in specificity]
        scalar = False
    for s in values:
        if not 0.0 < s < 1.0:
            raise ValueError(f"specificity values must be in (0, 1); got {s}.{percent_hint(s)}")
    if not values:
        raise ValueError("specificity needs at least one value in (0, 1).")
    return values, scalar


def _warn_missing(what: str, label: int) -> None:
    warnings.warn(
        f"Label {label} has no {what} samples; the metric is NaN for it.",
        UserWarning,
        stacklevel=3,
    )


def _finish(rows: list[list[float]], scalar: bool, binary: bool, like: Tensor) -> Tensor:
    out = torch.tensor(rows, dtype=torch.float64)  # (C, K)
    if scalar:
        out = out[:, 0]
    if binary:
        out = out[0]
    return out.to(device=like.device, dtype=_out_dtype(like))


def _threshold_from_roc(roc: _Roc, s: float) -> float:
    allowed = roc.fpr <= (1.0 - s) + _TOL
    index = int(torch.nonzero(allowed).flatten()[-1])
    return float(roc.thresholds[index])


def threshold_at_specificity(
    scores: Tensor,
    targets: Tensor,
    specificity: float | Sequence[float],
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Most sensitive threshold whose specificity on this data is at least ``specificity``.

    Among the ROC thresholds (the distinct scores and ``+inf``), return the smallest ``tau``
    with ``FPR(tau) <= 1 - specificity`` (with a 1e-9 tolerance, so that 0.9 with 10 negatives
    allows one false positive). Predicting *positive iff score >= tau* on this data then
    reaches at least the target specificity with the highest possible sensitivity.

    Parameters
    ----------
    scores : Tensor, shape (N,), (N, 1) or (N, C)
        Scores; higher means more likely positive.
    targets : Tensor, same shape as ``scores``
        Targets; ``> 0.5`` is positive, ``< 0.5`` negative, ``0.5`` ignored.
    specificity : float or sequence of K floats
        Target specificities in (0, 1).
    mask : Tensor, same shape as ``scores``, optional
        Non-zero elements are used.

    Returns
    -------
    Tensor, shape (), (K,), (C,) or (C, K)
        Thresholds in the space of ``scores`` (NaN for labels without negatives).

    Examples
    --------
    >>> import torch
    >>> from aucreshape.metrics import threshold_at_specificity
    >>> scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    >>> targets = torch.tensor([0, 0, 0, 1, 0, 1])
    >>> threshold_at_specificity(scores, targets, 0.75)  # predict positive iff score >= 0.4
    tensor(0.4000)
    """
    values, scalar = _specificities(specificity)
    pairs, binary = _labels(scores, targets, mask)
    rows = []
    for c, (s, pos) in enumerate(pairs):
        roc = _roc(s, pos)
        if roc.num_neg == 0:
            _warn_missing("negative", c)
            rows.append([math.nan] * len(values))
            continue
        rows.append([_threshold_from_roc(roc, v) for v in values])
    return _finish(rows, scalar, binary, scores)


def sensitivity_at_threshold(
    scores: Tensor,
    targets: Tensor,
    threshold: float | Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Fraction of positives with ``score >= threshold``.

    Parameters
    ----------
    scores : Tensor, shape (N,), (N, 1) or (N, C)
        Scores.
    targets : Tensor, same shape as ``scores``
        Targets; ``> 0.5`` is positive.
    threshold : float or Tensor
        For binary scores: a float, shape () or (K,) (or (1, K)). For C labels: a float,
        shape (), (C,) or (C, K), e.g. ``AUCReshapingLoss.thresholds``.
    mask : Tensor, same shape as ``scores``, optional
        Non-zero elements are used.

    Returns
    -------
    Tensor
        Sensitivities in [0, 1], shaped like ``threshold`` (per label for multi-label input;
        binary ``(1, K)`` thresholds give ``(K,)``). NaN for labels without positives.

    Examples
    --------
    >>> import torch
    >>> from aucreshape.metrics import sensitivity_at_threshold
    >>> scores = torch.tensor([0.1, 0.4, 0.6, 0.9])
    >>> targets = torch.tensor([1, 1, 1, 0])
    >>> sensitivity_at_threshold(scores, targets, 0.5)
    tensor(0.3333)
    """
    pairs, binary = _labels(scores, targets, mask)
    num_labels = len(pairs)
    tau = torch.as_tensor(threshold, dtype=torch.float64).detach().cpu()
    if binary:
        if tau.dim() == 2 and tau.shape[0] == 1:
            tau = tau[0]
        if tau.dim() > 1:
            raise ValueError(
                f"For binary scores, threshold must be a float or have shape (K,); "
                f"got {tuple(tau.shape)}."
            )
        table = tau.reshape(1, -1)
        out_shape: tuple[int, ...] = tuple(tau.shape)
    else:
        if tau.dim() == 0:
            table = tau.expand(num_labels, 1)
            out_shape = (num_labels,)
        elif tau.dim() in (1, 2) and tau.shape[0] == num_labels:
            table = tau.reshape(num_labels, -1)
            out_shape = tuple(tau.shape)
        else:
            raise ValueError(
                f"For {num_labels} labels, threshold must be a float or have shape "
                f"({num_labels},) or ({num_labels}, K); got {tuple(tau.shape)}."
            )
    rows = []
    for c, (s, pos) in enumerate(pairs):
        num_pos = int(pos.sum())
        if num_pos == 0:
            _warn_missing("positive", c)
            rows.append([math.nan] * table.shape[1])
            continue
        hits = (s[pos].unsqueeze(-1) >= table[c].unsqueeze(0)).sum(0).to(torch.float64)
        rows.append((hits / num_pos).tolist())
    out = torch.tensor(rows, dtype=torch.float64).reshape(out_shape)
    return out.to(device=scores.device, dtype=_out_dtype(scores))


def _out_dtype(like: Tensor) -> torch.dtype:
    return torch.float64 if like.dtype == torch.float64 else torch.float32


def _interpolated_tpr(roc: _Roc, x: float) -> float:
    i = int(torch.nonzero(roc.fpr <= x + _TOL).flatten()[-1])
    f0 = float(roc.fpr[i])
    if abs(f0 - x) <= _TOL or i == roc.fpr.numel() - 1:
        return float(roc.tpr[i])
    f1 = float(roc.fpr[i + 1])
    t0, t1 = float(roc.tpr[i]), float(roc.tpr[i + 1])
    return t0 + (t1 - t0) * (x - f0) / (f1 - f0)


def sensitivity_at_specificity(
    scores: Tensor,
    targets: Tensor,
    specificity: float | Sequence[float],
    *,
    interpolate: bool = True,
    mask: Tensor | None = None,
) -> Tensor:
    """Sensitivity read from the ROC curve at a given specificity.

    Parameters
    ----------
    scores : Tensor, shape (N,), (N, 1) or (N, C)
        Scores.
    targets : Tensor, same shape as ``scores``
        Targets; ``> 0.5`` is positive, ``< 0.5`` negative, ``0.5`` ignored.
    specificity : float or sequence of K floats
        Specificities in (0, 1).
    interpolate : bool, default True
        ``True``: TPR at FPR = 1 - specificity, interpolated linearly between ROC vertices
        (the top of a vertical segment when it is hit exactly), as is common practice.
        ``False``: the sensitivity of :func:`threshold_at_specificity`, i.e. what a real
        threshold achieves on this data.
    mask : Tensor, same shape as ``scores``, optional
        Non-zero elements are used.

    Returns
    -------
    Tensor, shape (), (K,), (C,) or (C, K)
        Sensitivities in [0, 1] (NaN for labels missing a class).

    Examples
    --------
    >>> import torch
    >>> from aucreshape.metrics import sensitivity_at_specificity
    >>> scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    >>> targets = torch.tensor([0, 0, 0, 1, 0, 1])
    >>> sensitivity_at_specificity(scores, targets, [0.75, 0.9])
    tensor([1.0000, 0.5000])
    """
    values, scalar = _specificities(specificity)
    pairs, binary = _labels(scores, targets, mask)
    rows = []
    for c, (s, pos) in enumerate(pairs):
        roc = _roc(s, pos)
        if roc.num_neg == 0 or roc.num_pos == 0:
            _warn_missing("negative" if roc.num_neg == 0 else "positive", c)
            rows.append([math.nan] * len(values))
            continue
        if interpolate:
            rows.append([_interpolated_tpr(roc, 1.0 - v) for v in values])
        else:
            rows.append(
                [float((s[pos] >= _threshold_from_roc(roc, v)).double().mean()) for v in values]
            )
    return _finish(rows, scalar, binary, scores)


def partial_auc(
    scores: Tensor,
    targets: Tensor,
    max_fpr: float = 1.0,
    *,
    standardized: bool = False,
    mask: Tensor | None = None,
) -> Tensor:
    """Area under the ROC curve for false-positive rates in ``[0, max_fpr]``.

    Parameters
    ----------
    scores : Tensor, shape (N,), (N, 1) or (N, C)
        Scores.
    targets : Tensor, same shape as ``scores``
        Targets; ``> 0.5`` is positive, ``< 0.5`` negative, ``0.5`` ignored.
    max_fpr : float, default 1.0
        Upper FPR bound in (0, 1]; 1 gives the full AUC.
    standardized : bool, default False
        ``False`` returns the raw area (at most ``max_fpr``). ``True`` applies the McClish
        correction ``0.5 * (1 + (A - A_min) / (A_max - A_min))``, which equals
        ``sklearn.metrics.roc_auc_score(..., max_fpr=max_fpr)``.
    mask : Tensor, same shape as ``scores``, optional
        Non-zero elements are used.

    Returns
    -------
    Tensor, shape () or (C,)
        Partial AUC values (NaN for labels missing a class).

    Examples
    --------
    >>> import torch
    >>> from aucreshape.metrics import partial_auc
    >>> scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    >>> targets = torch.tensor([0, 0, 0, 1, 0, 1])
    >>> partial_auc(scores, targets)
    tensor(0.8750)
    """
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in (0, 1]; got {max_fpr}.")
    pairs, binary = _labels(scores, targets, mask)
    rows = []
    for c, (s, pos) in enumerate(pairs):
        roc = _roc(s, pos)
        if roc.num_neg == 0 or roc.num_pos == 0:
            _warn_missing("negative" if roc.num_neg == 0 else "positive", c)
            rows.append([math.nan])
            continue
        fpr, tpr = roc.fpr, roc.tpr
        if max_fpr < 1.0:
            stop = int(
                torch.searchsorted(fpr, torch.tensor(max_fpr, dtype=torch.float64), right=True)
            )
            x0, x1 = float(fpr[stop - 1]), float(fpr[stop])
            y0, y1 = float(tpr[stop - 1]), float(tpr[stop])
            y_at = y0 + (y1 - y0) * (max_fpr - x0) / (x1 - x0) if x1 > x0 else y1
            fpr = torch.cat([fpr[:stop], torch.tensor([max_fpr], dtype=torch.float64)])
            tpr = torch.cat([tpr[:stop], torch.tensor([y_at], dtype=torch.float64)])
        area = float(torch.trapezoid(tpr, fpr))
        if standardized:
            min_area = 0.5 * max_fpr**2
            area = 0.5 * (1.0 + (area - min_area) / (max_fpr - min_area))
        rows.append([area])
    return _finish(rows, True, binary, scores)
