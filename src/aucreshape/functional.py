"""Stateless, compile-friendly AUCReshaping functions.

Every function here is pure: no state, no host synchronisation and no Python branching on
tensor values, so it can run inside ``torch.compile`` and CUDA graphs. Thresholds are in
logit space and listed from the lowest to the highest specificity.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from aucreshape._validation import (
    as_label_matrix,
    check_choice,
    check_nonnegative_floats,
    compute_dtype,
)

__all__ = [
    "BoostSchedule",
    "auc_reshaping_loss",
    "boost_values",
    "linear_ramp",
    "reshaping_weights",
]

BoostSchedule = Callable[[Tensor], Tensor]
"""A callable mapping a distance below the threshold (logit units, ``>= 0``) to a boost."""

Boosts = float | Sequence[float] | Tensor | BoostSchedule
Combine = Literal["band", "sum", "max"]
Mode = Literal["prob", "logit", "weight"]
Reduction = Literal["mean", "sum", "none"]

_COMBINES = ("band", "sum", "max")
_MODES = ("prob", "logit", "weight")
_REDUCTIONS = ("mean", "sum", "none")


def _thresholds_matrix(thresholds: Tensor, num_labels: int, like: Tensor) -> Tensor:
    """Validate thresholds and return them as ``(1 or C, K)`` on ``like``'s device/dtype."""
    if not isinstance(thresholds, Tensor):
        raise TypeError(
            f"thresholds must be a torch.Tensor in logit space; got {type(thresholds).__name__}."
        )
    if thresholds.dim() not in (1, 2):
        raise ValueError(
            f"thresholds must have shape (K,) or (C, K); got {tuple(thresholds.shape)}."
        )
    theta = thresholds if thresholds.dim() == 2 else thresholds.unsqueeze(0)
    if theta.shape[-1] == 0:
        raise ValueError("thresholds needs at least one threshold (K >= 1).")
    if theta.shape[0] not in (1, num_labels):
        raise ValueError(
            f"thresholds has {theta.shape[0]} rows but the logits have {num_labels} labels; "
            "pass shape (K,) to share thresholds or (C, K) with one row per label."
        )
    return theta.detach().to(device=like.device, dtype=like.dtype)


def _boost_matrix(
    boost: float | Sequence[float] | Tensor, theta: Tensor, num_labels: int
) -> Tensor:
    """Validate boosts and return them as a tensor broadcastable to ``theta`` ``(..., K)``."""
    k = theta.shape[-1]
    if isinstance(boost, Tensor):
        b = boost.detach().to(device=theta.device, dtype=theta.dtype)
        if b.dim() == 0:
            return b.expand(k)
        rows_ok = b.dim() == 1 or (b.dim() == 2 and b.shape[0] in (1, num_labels))
        if b.shape[-1] != k or not rows_ok:
            raise ValueError(
                f"boost must have shape (K,) or (C, K) = ({num_labels}, {k}); got {tuple(b.shape)}."
            )
        return b
    values = [float(boost)] * k if isinstance(boost, (int, float)) else [float(v) for v in boost]
    check_nonnegative_floats("boost", values)
    if len(values) != k:
        raise ValueError(f"Expected one boost per threshold (K={k}); got {len(values)}: {values}.")
    return torch.tensor(values, device=theta.device, dtype=theta.dtype)


def _boosts(
    z: Tensor,
    y: Tensor,
    m: Tensor | None,
    theta: Tensor,
    boost: Boosts,
    combine: Combine,
) -> Tensor:
    """Core of :func:`boost_values` on validated ``(N, C)`` inputs (detached ``z``)."""
    check_choice("combine", combine, _COMBINES)
    positive = y > 0.5
    if m is not None:
        positive = positive & m
    if callable(boost) and not isinstance(boost, Tensor):
        if theta.shape[-1] != 1:
            raise ValueError(
                "A boost schedule needs exactly one threshold (one target specificity); "
                f"got {theta.shape[-1]}."
            )
        threshold = theta[..., 0]
        below = z < threshold
        distance = torch.clamp(threshold - z, min=0.0)
        b = torch.clamp(boost(distance).to(z.dtype), min=0.0)
        return torch.where(positive & below, b, torch.zeros_like(z))

    table = _boost_matrix(boost, theta, z.shape[1])
    below = z.unsqueeze(-1) < theta.unsqueeze(0)  # (N, C, K)
    if combine == "band":
        # argmax returns the first maximal index, i.e. the lowest-specificity threshold the
        # positive falls below (a cumsum over this tiny last dimension is very slow on GPU).
        first = torch.argmax(below.to(torch.uint8), dim=-1, keepdim=True)
        b = torch.gather(table.expand(below.shape), -1, first).squeeze(-1) * below.any(dim=-1)
    elif combine == "sum":
        b = (below * table).sum(dim=-1)
    else:
        b = (below * table).amax(dim=-1)
    return torch.where(positive, b, torch.zeros_like(z))


def boost_values(
    logits: Tensor,
    targets: Tensor,
    thresholds: Tensor,
    boost: Boosts,
    *,
    combine: Combine = "band",
    mask: Tensor | None = None,
) -> Tensor:
    """Compute the per-element boost ``b_i`` of AUCReshaping.

    A positive (target ``> 0.5``) with logit ``z`` below threshold ``theta_k`` is
    *misclassified* at that threshold. With ``combine="band"`` it receives the boost of the
    lowest-specificity threshold it falls below, with ``"sum"`` the sum and with ``"max"``
    the maximum of the boosts of all thresholds it falls below. Negative elements
    (target ``< 0.5``), targets of exactly 0.5 and masked elements get 0.

    Parameters
    ----------
    logits : Tensor, shape (N,), (N, 1) or (N, C), floating point
        Raw model outputs (before the sigmoid).
    targets : Tensor, same shape as ``logits``
        Targets in ``[0, 1]``. Elements with target ``> 0.5`` are positives and can be boosted.
    thresholds : Tensor, shape (K,) or (C, K)
        Thresholds in logit space, from the lowest to the highest specificity. ``-inf``
        disables a threshold.
    boost : float, sequence of K floats, Tensor of shape (K,) or (C, K), or BoostSchedule
        Boost attached to each threshold, or a schedule ``f(distance)`` of the distance
        ``theta - z`` below a single threshold (then K must be 1). Schedule outputs are clamped
        at 0.
    combine : {"band", "sum", "max"}, default "band"
        How boosts combine when a positive falls below several thresholds.
    mask : Tensor, same shape as ``logits``, optional
        Non-zero elements are used; zero elements get no boost.

    Returns
    -------
    Tensor, same shape as ``logits``
        Detached, non-negative boosts in float32 (float64 for float64 logits).

    Examples
    --------
    >>> import torch
    >>> from aucreshape import boost_values
    >>> logits = torch.tensor([2.0, 0.5, -1.0, 0.5])
    >>> targets = torch.tensor([1.0, 1.0, 1.0, 0.0])
    >>> boost_values(logits, targets, torch.tensor([0.0, 1.0]), [0.1, 0.3])
    tensor([0.0000, 0.3000, 0.1000, 0.0000])
    """
    z, y, m = as_label_matrix(logits, targets, mask)
    z = z.detach().to(compute_dtype(logits.dtype))
    theta = _thresholds_matrix(thresholds, z.shape[1], z)
    b = _boosts(z, y.to(z.dtype), None if m is None else m != 0, theta, boost, combine)
    return b.reshape(logits.shape)


def reshaping_weights(
    logits: Tensor,
    targets: Tensor,
    thresholds: Tensor,
    boost: Boosts,
    *,
    combine: Combine = "band",
    mask: Tensor | None = None,
) -> Tensor:
    """Return per-element weights ``1 + b_i`` for use with any per-sample loss.

    Parameters
    ----------
    logits : Tensor, shape (N,), (N, 1) or (N, C), floating point
        Raw model outputs.
    targets : Tensor, same shape as ``logits``
        Targets in ``[0, 1]``.
    thresholds : Tensor, shape (K,) or (C, K)
        Thresholds in logit space, ascending in specificity.
    boost : float, sequence of K floats, Tensor of shape (K,) or (C, K), or BoostSchedule
        Boost attached to each threshold.
    combine : {"band", "sum", "max"}, default "band"
        How boosts combine across thresholds.
    mask : Tensor, same shape as ``logits``, optional
        Non-zero elements are used.

    Returns
    -------
    Tensor, same shape as ``logits``
        Detached weights ``>= 1``.

    Examples
    --------
    >>> import torch
    >>> from aucreshape import reshaping_weights
    >>> logits = torch.tensor([2.0, -1.0])
    >>> targets = torch.tensor([1.0, 1.0])
    >>> reshaping_weights(logits, targets, torch.tensor([0.0]), 0.5)
    tensor([1.0000, 1.5000])
    """
    return 1.0 + boost_values(logits, targets, thresholds, boost, combine=combine, mask=mask)


def _pos_weight(pos_weight: float | Tensor | None, like: Tensor) -> float | Tensor | None:
    if pos_weight is None:
        return None
    if not isinstance(pos_weight, Tensor):
        return float(pos_weight)  # a Python scalar: no host-to-device copy
    num_labels = like.shape[1]
    if pos_weight.dim() > 1 or pos_weight.numel() not in (1, num_labels):
        raise ValueError(
            f"pos_weight must be a float or a tensor of shape (C,) = ({num_labels},); "
            f"got shape {tuple(pos_weight.shape)}."
        )
    return pos_weight.to(device=like.device, dtype=like.dtype).reshape(-1)


def _reshaped_loss(
    logits: Tensor,
    targets: Tensor,
    thresholds: Tensor,
    boost: Boosts,
    *,
    mode: Mode,
    combine: Combine,
    pos_weight: float | Tensor | None,
    weight: Tensor | None,
    mask: Tensor | None,
    reduction: Reduction,
    min_ratio: float,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Compute the loss; also return the boosts and boolean mask as ``(N, C)`` matrices."""
    check_choice("mode", mode, _MODES)
    check_choice("reduction", reduction, _REDUCTIONS)
    if not 0.0 < min_ratio < 1.0:
        raise ValueError(f"min_ratio must be in (0, 1); got {min_ratio}.")
    z_in, y_in, m_in = as_label_matrix(logits, targets, mask)
    dtype = compute_dtype(logits.dtype)
    z = z_in.to(dtype)
    y = y_in.to(dtype)
    m = None if m_in is None else m_in != 0
    if m is not None:
        y = torch.where(m, y, torch.zeros_like(y))  # NaN targets under the mask are harmless
    theta = _thresholds_matrix(thresholds, z.shape[1], z)
    pw = _pos_weight(pos_weight, z)
    b = _boosts(z.detach(), y, m, theta, boost, combine)

    log_p = F.logsigmoid(z)
    log_q = F.logsigmoid(-z)
    if mode == "prob":
        # log(p - b') = log p + log1p(-b'/p) with b' = c * p.detach() and c = min(b/p, 1 - r).
        # c is computed in log space and the live ratio b'/p = c * exp(log p_det - log p), so
        # the value is finite for every finite logit and the gradient is (1 - p)/(1 - c).
        log_p_det = log_p.detach()
        log_c = torch.clamp(torch.log(b) - log_p_det, max=math.log1p(-min_ratio))
        ratio = torch.exp(log_c) * torch.exp(log_p_det - log_p)
        pos_term = log_p + torch.log1p(-ratio)
    elif mode == "logit":
        pos_term = F.logsigmoid(z - b)
    else:
        pos_term = log_p
    if pw is not None:
        pos_term = pw * pos_term
    loss = -(y * pos_term + (1.0 - y) * log_q)
    if mode == "weight":
        loss = loss * (1.0 + b)
    if m is not None:
        loss = torch.where(m, loss, torch.zeros_like(loss))
    loss = loss.reshape(logits.shape)
    if weight is not None:
        loss = loss * weight.to(device=loss.device, dtype=dtype)

    if reduction == "none":
        return loss, b, m
    total = loss.sum()
    if reduction == "sum":
        return total, b, m
    if m is None:
        return total / max(loss.numel(), 1), b, m
    return total / m.sum().clamp(min=1).to(dtype), b, m


def auc_reshaping_loss(
    logits: Tensor,
    targets: Tensor,
    thresholds: Tensor,
    boost: Boosts,
    *,
    mode: Mode = "prob",
    combine: Combine = "band",
    pos_weight: float | Tensor | None = None,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    reduction: Reduction = "mean",
    min_ratio: float = 0.1,
) -> Tensor:
    """AUCReshaping loss: binary cross-entropy with boosted misclassified positives.

    With ``p = sigmoid(z)``, ``w`` the positive weight and ``b`` from :func:`boost_values`::

        prob   : -[w*y*log(p - b') + (1 - y)*log(1 - p)],  b' = min(b, (1-min_ratio)*p.detach())
        logit  : -[w*y*log(sigmoid(z - b)) + (1 - y)*log(sigmoid(-z))]
        weight : (1 + b) * BCE(z, y)

    ``prob`` is Eq. 1 of the paper. The cap ``b'`` keeps the logarithm defined and the
    gradient alive: it is at most ``1/min_ratio`` times the BCE gradient and never zero.
    With ``b = 0`` every mode equals ``BCEWithLogitsLoss``. No host synchronisation happens.

    Parameters
    ----------
    logits : Tensor, shape (N,), (N, 1) or (N, C), floating point
        Raw model outputs.
    targets : Tensor, same shape as ``logits``
        Targets in ``[0, 1]`` (values are not checked, as in ``BCEWithLogitsLoss``).
    thresholds : Tensor, shape (K,) or (C, K)
        Thresholds in logit space, ascending in specificity.
    boost : float, sequence of K floats, Tensor of shape (K,) or (C, K), or BoostSchedule
        Boost attached to each threshold.
    mode : {"prob", "logit", "weight"}, default "prob"
        How the boost enters the loss: probability offset, logit margin or loss weight.
    combine : {"band", "sum", "max"}, default "band"
        How boosts combine across thresholds.
    pos_weight : float or Tensor of shape (C,), optional
        Weight of the positive term, as in ``BCEWithLogitsLoss``.
    weight : Tensor broadcastable to ``logits``, optional
        Manual rescaling of each element's loss, as in ``BCEWithLogitsLoss``.
    mask : Tensor, same shape as ``logits``, optional
        Non-zero elements are used. Masked elements contribute exactly 0 loss and gradient,
        even if their targets are NaN.
    reduction : {"mean", "sum", "none"}, default "mean"
        ``"mean"`` divides the sum by the number of unmasked elements; an empty or fully
        masked batch gives 0.
    min_ratio : float, default 0.1
        Floor of ``prob`` mode (ignored by the other modes): the boost is capped so that
        ``p - b >= min_ratio * p``.

    Returns
    -------
    Tensor
        Scalar for ``"mean"``/``"sum"``; same shape as ``logits`` for ``"none"`` (0 where
        masked). float64 for float64 logits, float32 otherwise (half-precision inputs are
        computed and returned in float32, as ``BCEWithLogitsLoss`` does under autocast).

    Examples
    --------
    >>> import torch
    >>> from aucreshape import auc_reshaping_loss
    >>> logits = torch.tensor([2.0, 0.5, -1.0, 0.5])
    >>> targets = torch.tensor([1.0, 1.0, 1.0, 0.0])
    >>> auc_reshaping_loss(logits, targets, torch.tensor([0.0, 1.0]), [0.1, 0.3],
    ...                    reduction="none")
    tensor([0.1269, 1.1318, 1.7782, 0.9741])
    """
    loss, _, _ = _reshaped_loss(
        logits,
        targets,
        thresholds,
        boost,
        mode=mode,
        combine=combine,
        pos_weight=pos_weight,
        weight=weight,
        mask=mask,
        reduction=reduction,
        min_ratio=min_ratio,
    )
    return loss


def linear_ramp(slope: float, max_boost: float) -> BoostSchedule:
    """Build the schedule ``b(d) = min(slope * d, max_boost)``.

    This is the paper's "continuous" variant: the boost grows with the distance ``d`` (in
    logit units) of a misclassified positive below the operating-point threshold, and is 0
    at the threshold itself.

    Parameters
    ----------
    slope : float
        Boost per logit unit of distance below the threshold. Must be ``>= 0``.
    max_boost : float
        Upper bound of the boost. Must be ``>= 0``.

    Returns
    -------
    BoostSchedule
        A callable mapping distances (Tensor, any shape) to boosts of the same shape.

    Examples
    --------
    >>> import torch
    >>> from aucreshape import linear_ramp
    >>> linear_ramp(0.5, 0.4)(torch.tensor([0.0, 0.5, 2.0]))
    tensor([0.0000, 0.2500, 0.4000])
    """
    if not (math.isfinite(slope) and slope >= 0):
        raise ValueError(f"slope must be finite and >= 0; got {slope}.")
    if not (math.isfinite(max_boost) and max_boost >= 0):
        raise ValueError(f"max_boost must be finite and >= 0; got {max_boost}.")

    def schedule(distance: Tensor) -> Tensor:
        return torch.clamp(distance * slope, max=max_boost)

    return schedule
