"""The drop-in ``AUCReshapingLoss`` module."""

from __future__ import annotations

import math
import warnings
import weakref
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from aucreshape._validation import (
    as_label_matrix,
    check_choice,
    check_nonnegative_floats,
    check_specificities,
)
from aucreshape.functional import (
    _COMBINES,
    _MODES,
    _REDUCTIONS,
    BoostSchedule,
    Combine,
    Mode,
    Reduction,
    _reshaped_loss,
)
from aucreshape.thresholds import Strategy, ThresholdTracker

__all__ = ["AUCReshapingLoss"]

_TRACKER_DEFAULTS = {
    "strategy": "window",
    "window_size": None,
    "momentum": 0.9,
    "min_negatives": None,
    "max_samples": None,
    "sync_distributed": True,
    "initial_thresholds": None,
}


class _Sentinel:
    """Lives as long as a loss's autograd graph; reports if the graph dies unused."""

    __slots__ = ("criterion", "fired")

    def __init__(self, criterion: AUCReshapingLoss) -> None:
        self.criterion = weakref.ref(criterion)
        self.fired = False

    def __del__(self) -> None:
        criterion = self.criterion()
        if not self.fired and criterion is not None:
            criterion._no_backward()


class _BackwardSentinel(torch.autograd.Function):
    """Identity whose context holds a sentinel; backward marks it as used."""

    @staticmethod
    def forward(ctx: Any, loss: Tensor, sentinel: _Sentinel) -> Tensor:  # noqa: ANN401
        ctx.sentinel = sentinel
        return loss.view_as(loss)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:  # noqa: ANN401
        ctx.sentinel.fired = True
        return grad, None


class AUCReshapingLoss(nn.Module):
    """Drop-in replacement for ``BCEWithLogitsLoss`` that applies AUCReshaping.

    In training (``criterion.training`` and gradients enabled), positives whose logit falls
    below the high-specificity threshold get a boosted loss; the thresholds come from an
    internal :class:`~aucreshape.ThresholdTracker`, which is updated *after* the loss of each
    batch is computed. Outside training (``criterion.eval()`` or ``torch.no_grad()``) the
    loss is plain binary cross-entropy and nothing changes, so validation losses stay
    comparable with a BCE baseline.

    Parameters
    ----------
    target_specificity : float or sequence of float, optional
        Target specificities in (0, 1), strictly ascending. Defaults to 0.95, or to the
        tracker's when ``tracker`` is given.
    boost : float, sequence of float, Tensor of shape (K,) or (C, K), or BoostSchedule
        Boost per specificity (a float is broadcast), per label and specificity, or a
        distance schedule such as :func:`~aucreshape.linear_ramp` (one specificity only).
        Default 0.1.
    mode : {"prob", "logit", "weight"}, default "prob"
        How the boost enters the loss (``"prob"`` is the paper's Eq. 1).
    combine : {"band", "sum", "max"}, default "band"
        How boosts combine across thresholds.
    num_labels : int, optional
        Number of labels C. Defaults to 1, or to the tracker's.
    pos_weight : float or Tensor of shape (C,), optional
        As in ``BCEWithLogitsLoss``.
    weight : Tensor broadcastable to the logits, optional
        As in ``BCEWithLogitsLoss``.
    reduction : {"mean", "sum", "none"}, default "mean"
        Reduction over unmasked elements.
    strategy : {"fixed", "batch", "window", "ema", "epoch"}, default "window"
        Threshold strategy of the internal tracker.
    tracker : ThresholdTracker, optional
        Use this tracker instead of building one (e.g. from
        :meth:`ThresholdTracker.from_scores`).
    window_size, momentum, min_negatives, max_samples, sync_distributed, initial_thresholds
        Forwarded to :class:`~aucreshape.ThresholdTracker`.
    auto_update : bool, optional
        ``None`` updates the tracker in training only; ``True`` on every call; ``False``
        never (call :meth:`update` yourself).
    warmup_steps : int, default 0
        Training calls (micro-batches) before boosting starts. The tracker still updates.
    min_ratio : float, default 0.1
        Floor of ``prob`` mode (ignored by the other modes).

    Attributes
    ----------
    tracker : ThresholdTracker
        The threshold tracker.
    last_stats : dict of str to Tensor
        Detached statistics of the last training call: ``boosted_fraction`` (C,),
        ``below_fraction`` (C, K), ``mean_boost`` (C,), ``thresholds`` (C, K),
        ``ready`` (C,) and ``step`` ().

    Examples
    --------
    >>> import torch
    >>> from aucreshape import AUCReshapingLoss
    >>> criterion = AUCReshapingLoss(0.9, boost=0.2, initial_thresholds=[0.0])
    >>> logits = torch.tensor([2.0, -1.0, 0.5, -2.0], requires_grad=True)
    >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    >>> loss = criterion(logits, targets)
    >>> loss.backward()
    >>> criterion.last_stats["boosted_fraction"]
    tensor([0.5000])
    """

    tracker: ThresholdTracker
    step: Tensor
    pos_weight: Tensor | None
    weight: Tensor | None
    boost_table: Tensor

    def __init__(
        self,
        target_specificity: float | Sequence[float] | None = None,
        boost: float | Sequence[float] | Tensor | BoostSchedule = 0.1,
        *,
        mode: Mode = "prob",
        combine: Combine = "band",
        num_labels: int | None = None,
        pos_weight: float | Tensor | None = None,
        weight: Tensor | None = None,
        reduction: Reduction = "mean",
        strategy: Strategy = "window",
        tracker: ThresholdTracker | None = None,
        window_size: int | None = None,
        momentum: float = 0.9,
        min_negatives: int | None = None,
        max_samples: int | None = None,
        sync_distributed: bool = True,
        initial_thresholds: Tensor | Sequence[float] | None = None,
        auto_update: bool | None = None,
        warmup_steps: int = 0,
        min_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        check_choice("mode", mode, _MODES)
        check_choice("combine", combine, _COMBINES)
        check_choice("reduction", reduction, _REDUCTIONS)
        if not 0.0 < min_ratio < 1.0:
            raise ValueError(f"min_ratio must be in (0, 1); got {min_ratio}.")
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError(f"warmup_steps must be a non-negative integer; got {warmup_steps!r}.")

        if tracker is None:
            tracker = ThresholdTracker(
                0.95 if target_specificity is None else target_specificity,
                num_labels=1 if num_labels is None else num_labels,
                strategy=strategy,
                window_size=window_size,
                momentum=momentum,
                min_negatives=min_negatives,
                max_samples=max_samples,
                sync_distributed=sync_distributed,
                initial_thresholds=initial_thresholds,
            )
        else:
            self._check_tracker(
                tracker,
                target_specificity,
                num_labels,
                {
                    "strategy": strategy,
                    "window_size": window_size,
                    "momentum": momentum,
                    "min_negatives": min_negatives,
                    "max_samples": max_samples,
                    "sync_distributed": sync_distributed,
                    "initial_thresholds": initial_thresholds,
                },
            )
        self.tracker = tracker
        k = len(tracker.target_specificity)
        c = tracker.num_labels

        self._schedule: BoostSchedule | None = None
        self._boost_values: list[float] | None = None
        if isinstance(boost, Tensor):
            table = boost.detach().float()
            if table.dim() == 0:
                table = table.expand(k)
            rows_ok = table.dim() == 1 or (table.dim() == 2 and table.shape[0] in (1, c))
            if table.shape[-1] != k or not rows_ok:
                raise ValueError(
                    f"boost tensor must have shape ({k},) or ({c}, {k}) for {k} "
                    f"specificities and {c} labels; got {tuple(boost.shape)}."
                )
            self.register_buffer("boost_table", table.clone(), persistent=False)
        elif callable(boost):
            if k != 1:
                raise ValueError(
                    "A boost schedule needs exactly one target_specificity (the operating "
                    f"point); got {k}: {list(tracker.target_specificity)}."
                )
            self._schedule = boost
        else:
            values = [float(boost)] * k if isinstance(boost, (int, float)) else list(boost)
            values = [float(v) for v in values]
            check_nonnegative_floats("boost", values)
            if len(values) != k:
                raise ValueError(
                    f"boost needs one value per target_specificity ({k}); got {len(values)}: "
                    f"{values}. Pass a single float to use the same boost everywhere."
                )
            self._boost_values = values
            # Stored once as a buffer so that forward never copies boosts to the device.
            self.register_buffer("boost_table", torch.tensor(values), persistent=False)

        self._pos_weight_value: float | None = None
        if isinstance(pos_weight, Tensor):
            self.register_buffer("pos_weight", pos_weight.detach().clone(), persistent=False)
        else:
            self.pos_weight = None
            self._pos_weight_value = None if pos_weight is None else float(pos_weight)
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone(), persistent=False)
        else:
            self.weight = None

        self.mode: Mode = mode
        self.combine: Combine = combine
        self.reduction: Reduction = reduction
        self.auto_update = auto_update
        self.warmup_steps = warmup_steps
        self.min_ratio = min_ratio
        self.register_buffer("step", torch.zeros((), dtype=torch.int64))
        self._stats: dict[str, Tensor] = {
            "boosted_fraction": torch.zeros(c),
            "below_fraction": torch.zeros(c, k),
            "mean_boost": torch.zeros(c),
            "thresholds": tracker.thresholds,
            "ready": tracker.ready,
            "step": torch.zeros((), dtype=torch.int64),
        }
        self._pending: tuple[Tensor, ...] | None = None
        self._warned_no_backward = False

    @staticmethod
    def _check_tracker(
        tracker: ThresholdTracker,
        target_specificity: float | Sequence[float] | None,
        num_labels: int | None,
        tracker_args: dict[str, object],
    ) -> None:
        changed = [
            name
            for name, value in tracker_args.items()
            if (
                value is not None
                if name == "initial_thresholds"
                else value != _TRACKER_DEFAULTS[name]
            )
        ]
        if changed:
            raise ValueError(
                f"Got a tracker and tracker arguments {changed}; configure the tracker itself "
                "and leave these arguments at their defaults."
            )
        if target_specificity is not None and (
            check_specificities(target_specificity) != tracker.target_specificity
        ):
            raise ValueError(
                f"target_specificity {target_specificity} differs from the tracker's "
                f"{list(tracker.target_specificity)}; omit it to use the tracker's."
            )
        if num_labels is not None and num_labels != tracker.num_labels:
            raise ValueError(
                f"num_labels={num_labels} differs from the tracker's {tracker.num_labels}; "
                "omit it to use the tracker's."
            )

    @property
    def thresholds(self) -> Tensor:
        """Tensor, shape (C, K): the tracker's thresholds in logit space."""
        return self.tracker.thresholds

    def _boost(self) -> Tensor | BoostSchedule:
        if self._schedule is not None:
            return self._schedule
        table: Tensor = self.boost_table
        return table

    def _check_labels(self, logits: Tensor) -> None:
        c = 1 if logits.dim() == 1 else logits.shape[-1]
        if logits.dim() in (1, 2) and c != self.tracker.num_labels:
            raise ValueError(
                f"AUCReshapingLoss was built with num_labels={self.tracker.num_labels} but the "
                f"logits have {c} labels (shape {tuple(logits.shape)}); pass num_labels={c}."
            )

    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        """Compute the loss with the current thresholds, then update the tracker.

        Parameters
        ----------
        logits : Tensor, shape (N,), (N, 1) or (N, C)
            Raw model outputs.
        targets : Tensor, same shape as ``logits``
            Targets in ``[0, 1]``.
        mask : Tensor, same shape as ``logits``, optional
            Non-zero elements are used; masked targets may hold any value, even NaN.

        Returns
        -------
        Tensor
            Scalar for ``"mean"``/``"sum"``; same shape as ``logits`` for ``"none"``.
            float64 for float64 logits, float32 otherwise.
        """
        as_label_matrix(logits, targets, mask)
        self._check_labels(logits)
        if self.step.device != logits.device:
            self.to(logits.device)
        pos_weight = self.pos_weight if self.pos_weight is not None else self._pos_weight_value
        training = self.training and torch.is_grad_enabled()
        if not training:
            loss, _, _ = _reshaped_loss(
                logits,
                targets,
                torch.full((1,), -math.inf, device=logits.device),
                0.0,
                mode="prob",
                combine="band",
                pos_weight=pos_weight,
                weight=self.weight,
                mask=mask,
                reduction=self.reduction,
                min_ratio=self.min_ratio,
            )
            if self.auto_update:
                self.tracker.update(logits, targets, mask)
            return loss

        active = self.tracker.active_thresholds()
        if self.warmup_steps > 0:
            warm = self.step < self.warmup_steps
            used = torch.where(warm, torch.full_like(active, -math.inf), active)
        else:
            used = active
        loss, b, m = _reshaped_loss(
            logits,
            targets,
            used,
            self._boost(),
            mode=self.mode,
            combine=self.combine,
            pos_weight=pos_weight,
            weight=self.weight,
            mask=mask,
            reduction=self.reduction,
            min_ratio=self.min_ratio,
        )
        self._record_stats(logits, targets, m, b, active)
        self.step.add_(1)
        if self.auto_update is not False:
            self.tracker.update(logits, targets, mask)
        return self._watch_backward(loss)

    def _watch_backward(self, loss: Tensor) -> Tensor:
        """Attach a sentinel that warns once if this loss's graph is freed without backward().

        That is the signature of a validation pass with gradients enabled (``model.eval()``
        does not reach a standalone criterion), which silently feeds validation data to the
        tracker. Losses summed before one ``backward()`` keep their graph alive until then,
        so they do not warn. No host synchronisation.
        """
        if self._warned_no_backward or not loss.requires_grad:
            return loss
        out: Tensor = _BackwardSentinel.apply(loss, _Sentinel(self))
        return out

    def _no_backward(self) -> None:
        if self._warned_no_backward:
            return
        self._warned_no_backward = True
        warnings.warn(
            "AUCReshapingLoss: a loss computed in training mode was discarded without "
            "backward(). If this was a validation pass, it updated the threshold tracker with "
            "validation data and returned the boosted loss: model.eval() does not switch the "
            "criterion. Wrap validation in torch.no_grad() (or call criterion.eval() and later "
            "criterion.train()).",
            UserWarning,
            stacklevel=2,
        )

    def _record_stats(
        self, logits: Tensor, targets: Tensor, m: Tensor | None, b: Tensor, active: Tensor
    ) -> None:
        """Keep what ``last_stats`` needs; the reductions run only when it is read."""
        z, y, _ = as_label_matrix(logits, targets, None)
        positive = y.detach() > 0.5
        if m is not None:
            positive = positive & m
        self._pending = (
            z.detach(),
            positive,
            b,
            active,
            self.tracker.thresholds,
            self.tracker.ready,
            self.step + 1,
        )

    @property
    def last_stats(self) -> dict[str, Tensor]:
        """Statistics of the last training call (detached tensors, computed on first read).

        ``boosted_fraction`` (C,), ``below_fraction`` (C, K), ``mean_boost`` (C,),
        ``thresholds`` (C, K), ``ready`` (C,) and ``step`` (). No host synchronisation.
        """
        if self._pending is not None:
            z, positive, b, active, thresholds, ready, step = self._pending
            with torch.no_grad():
                num_pos = positive.sum(0).clamp(min=1).float()
                boosted = b > 0
                num_boosted = boosted.sum(0)
                below = (z.float().unsqueeze(-1) < active.unsqueeze(0)) & positive.unsqueeze(-1)
                self._stats = {
                    "boosted_fraction": num_boosted.float() / num_pos,
                    "below_fraction": below.sum(0).float() / num_pos.unsqueeze(-1),
                    "mean_boost": b.sum(0).float() / num_boosted.clamp(min=1).float(),
                    "thresholds": thresholds,
                    "ready": ready,
                    "step": step,
                }
            self._pending = None
        return self._stats

    def update(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> None:
        """Update the tracker manually (for ``auto_update=False``).

        Parameters
        ----------
        logits : Tensor, shape (N,), (N, 1) or (N, C)
            Raw model outputs; detached internally.
        targets : Tensor, same shape as ``logits``
            Targets in ``[0, 1]``.
        mask : Tensor, same shape as ``logits``, optional
            Non-zero elements are used.
        """
        self._check_labels(logits)
        self.tracker.update(logits, targets, mask)

    def end_epoch(self) -> None:
        """Delegate to :meth:`ThresholdTracker.end_epoch` (finalises ``strategy="epoch"``)."""
        self.tracker.end_epoch()

    def extra_repr(self) -> str:
        """Summarise the configuration for ``print(criterion)``."""
        boost = "schedule" if self._schedule is not None else self._boost_values or "tensor"
        return (
            f"boost={boost}, mode={self.mode!r}, combine={self.combine!r}, "
            f"reduction={self.reduction!r}"
        )
