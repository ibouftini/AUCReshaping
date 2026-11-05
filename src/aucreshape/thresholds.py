"""Threshold tracking at a target specificity.

A :class:`ThresholdTracker` estimates, for every label, the logit thresholds at which the
negative (reference) class reaches the target specificities. All state lives in buffers, so
it is saved with ``state_dict``; updates are vectorised and free of host synchronisation on a
single process.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from aucreshape import metrics
from aucreshape._distributed import all_gather_rows, is_distributed
from aucreshape._validation import (
    as_label_matrix,
    check_choice,
    check_positive_int,
    check_specificities,
)

__all__ = ["Strategy", "ThresholdTracker"]

Strategy = Literal["fixed", "batch", "window", "ema", "epoch"]
_STRATEGIES = ("fixed", "batch", "window", "ema", "epoch")
_READY_CHECK_AT = 1000  # one readiness check (a host sync) at this update count
_MASK32 = 0xFFFFFFFF


def _hash32(x: Tensor) -> Tensor:
    """Hash non-negative int64 values below 2**32 to [0, 2**32) without int64 overflow."""
    x = x & _MASK32
    x = ((x >> 16) ^ x) * 0x45D9F3B & _MASK32
    x = ((x >> 16) ^ x) * 0x45D9F3B & _MASK32
    return (x >> 16) ^ x


def _sorted_quantile(values: Tensor, count: Tensor, q: Tensor) -> Tensor:
    """Linear-interpolation quantiles of the first ``count`` values of each row.

    Parameters
    ----------
    values : Tensor, shape (C, M)
        Values, padded with ``+inf`` beyond each row's count (so they sort last).
    count : Tensor, shape (C,), int64
        Number of valid values per row.
    q : Tensor, shape (K,)
        Quantile levels in (0, 1).

    Returns
    -------
    Tensor, shape (C, K)
        Quantiles (float32), ``-inf`` for rows without values.
    """
    if values.shape[1] == 0:
        return torch.full((values.shape[0], q.shape[0]), -math.inf, device=values.device)
    srt = values.float().sort(dim=1).values
    last = (count - 1).clamp(min=0).unsqueeze(1)  # (C, 1)
    pos = q.unsqueeze(0) * last.float()  # (C, K)
    lo = pos.floor().long()
    hi = torch.minimum(lo + 1, last)
    frac = pos - lo.float()
    v_lo = srt.gather(1, lo)
    v_hi = srt.gather(1, hi)
    out = v_lo + frac * (v_hi - v_lo)
    out = torch.where(frac == 0, v_lo, out)  # avoids inf - inf when both ends are padding
    return torch.where(count.unsqueeze(1) > 0, out, torch.full_like(out, -math.inf))


class ThresholdTracker(nn.Module):
    """Estimate the logit thresholds at one or more target specificities during training.

    For label ``c`` and specificity ``s``, the threshold is the ``s``-quantile of the logits
    of negatives (targets ``< 0.5``): a fraction ``s`` of negatives score below it. Positives
    below it are misclassified at that operating point.

    Parameters
    ----------
    target_specificity : float or sequence of float
        Target specificities in (0, 1), strictly ascending. K values.
    num_labels : int, default 1
        Number of labels C (1 for binary classification).
    strategy : {"fixed", "batch", "window", "ema", "epoch"}, default "window"
        How thresholds are estimated (see the table in the user guide):
        ``fixed`` never updates; ``batch`` uses the current batch; ``window`` a FIFO buffer
        of recent negatives; ``ema`` an exponential moving average of batch quantiles;
        ``epoch`` a uniform reservoir of the epoch, finalised by :meth:`end_epoch`.
    window_size : int, optional
        Buffer capacity per label for ``window``. Default ``max(4096, 2 * min_negatives)``.
    momentum : float, default 0.9
        EMA momentum for ``ema``, in [0, 1).
    min_negatives : int, optional
        Negatives a label's estimate needs before it is used. Default
        ``ceil(10 / (1 - max(target_specificity)))``.
    max_samples : int, optional
        Reservoir capacity per label for ``epoch``. Default ``max(65536, 2 * min_negatives)``.
    sync_distributed : bool, default True
        All-gather negatives' logits across ranks when ``torch.distributed`` is initialised.
    initial_thresholds : Tensor or sequence of float, shape (K,) or (C, K), optional
        Thresholds in logit space used until the strategy's own estimate is valid (a warm
        start). Required for ``fixed`` (or use :meth:`from_scores`).
    seed : int, default 0
        Seed of the reservoir sampler of ``epoch``.

    Examples
    --------
    >>> import torch
    >>> from aucreshape import ThresholdTracker
    >>> tracker = ThresholdTracker(0.9, min_negatives=10)
    >>> tracker.update(torch.arange(100.0), torch.zeros(100))
    >>> tracker.thresholds
    tensor([[89.1000]])
    """

    initial: Tensor
    seeded: Tensor
    estimate: Tensor
    support: Tensor
    """Tensor, shape (C,), int64: negatives behind each label's own estimate."""
    total_seen: Tensor
    buffer: Tensor
    buffer_count: Tensor
    _levels: Tensor

    def __init__(
        self,
        target_specificity: float | Sequence[float],
        *,
        num_labels: int = 1,
        strategy: Strategy = "window",
        window_size: int | None = None,
        momentum: float = 0.9,
        min_negatives: int | None = None,
        max_samples: int | None = None,
        sync_distributed: bool = True,
        initial_thresholds: Tensor | Sequence[float] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        specs = check_specificities(target_specificity)
        check_choice("strategy", strategy, _STRATEGIES)
        check_positive_int("num_labels", num_labels)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1); got {momentum}.")
        if min_negatives is None:
            min_negatives = math.ceil(10.0 / (1.0 - max(specs)) - 1e-9)
        check_positive_int("min_negatives", min_negatives)
        if strategy == "fixed" and initial_thresholds is None:
            raise ValueError(
                "strategy='fixed' needs initial_thresholds (logit space, shape (K,) or (C, K)); "
                "or build the tracker from scores with ThresholdTracker.from_scores(...)."
            )
        capacity = 0
        if strategy == "window":
            capacity = self._capacity("window_size", window_size, 4096, min_negatives, specs)
        elif strategy == "epoch":
            capacity = self._capacity("max_samples", max_samples, 65536, min_negatives, specs)

        self._specs = specs
        self._num_labels = num_labels
        self._strategy: Strategy = strategy
        self._momentum = float(momentum)
        self._min_negatives = min_negatives
        self._capacity_value = capacity
        self._seed = int(seed)
        self.sync_distributed = sync_distributed
        self._num_updates = 0
        self._warned = False

        c, k = num_labels, len(specs)
        initial = torch.full((c, k), -math.inf)
        seeded = torch.zeros(c, dtype=torch.bool)
        if initial_thresholds is not None:
            init = torch.as_tensor(initial_thresholds, dtype=torch.float32).detach().cpu()
            if init.dim() == 1:
                init = init.unsqueeze(0)
            if init.dim() != 2 or init.shape[1] != k or init.shape[0] not in (1, c):
                raise ValueError(
                    f"initial_thresholds must have shape ({k},) or ({c}, {k}) for "
                    f"{k} specificities and {c} labels; got {tuple(init.shape)}."
                )
            initial = init.expand(c, k).clone()
            seeded = torch.ones(c, dtype=torch.bool)
        self.register_buffer("initial", initial)
        self.register_buffer("seeded", seeded)
        self.register_buffer("estimate", torch.full((c, k), -math.inf))
        self.register_buffer("support", torch.zeros(c, dtype=torch.int64))
        if strategy != "fixed":
            self.register_buffer("total_seen", torch.zeros(c, dtype=torch.int64))
        if capacity:
            self.register_buffer("buffer", torch.full((c, capacity + 1), math.inf))
            self.register_buffer("buffer_count", torch.zeros(c, dtype=torch.int64))
        self.register_buffer("_levels", torch.tensor(specs, dtype=torch.float32), persistent=False)

    @staticmethod
    def _capacity(
        name: str, value: int | None, default: int, min_negatives: int, specs: tuple[float, ...]
    ) -> int:
        if value is None:
            return max(default, 2 * min_negatives)
        check_positive_int(name, value)
        if value < min_negatives:
            raise ValueError(
                f"{name}={value} is smaller than min_negatives={min_negatives}, the number of "
                f"negatives needed for target_specificity={max(specs)}; the tracker would "
                f"never be ready. Use {name} >= {min_negatives} (e.g. {2 * min_negatives}), "
                "or leave it unset for an automatic size."
            )
        return value

    # ----------------------------------------------------------------------------- properties

    @property
    def target_specificity(self) -> tuple[float, ...]:
        """Tuple of K floats: the target specificities, ascending."""
        return self._specs

    @property
    def num_labels(self) -> int:
        """Number of labels C."""
        return self._num_labels

    @property
    def strategy(self) -> Strategy:
        """The threshold strategy."""
        return self._strategy

    @property
    def min_negatives(self) -> int:
        """Negatives needed before a label's own estimate is used."""
        return self._min_negatives

    @property
    def capacity(self) -> int:
        """Buffer size per label of the ``window``/``epoch`` strategies (0 otherwise)."""
        return self._capacity_value

    def _own_ready(self) -> Tensor:
        if self._strategy == "fixed":
            return torch.zeros_like(self.seeded)
        support: Tensor = self.support
        return support >= self._min_negatives

    @property
    def thresholds(self) -> Tensor:
        """Tensor, shape (C, K): current thresholds in logit space (``-inf`` if unknown).

        The strategy's own estimate where it is valid, otherwise the initial thresholds.
        """
        own = self._own_ready().unsqueeze(1)
        estimate: Tensor = self.estimate
        initial: Tensor = self.initial
        return torch.where(own, estimate.float(), initial.float())

    @property
    def ready(self) -> Tensor:
        """Tensor, shape (C,), bool: whether boosting is allowed for each label."""
        seeded: Tensor = self.seeded
        if self._strategy == "fixed":
            return torch.ones_like(seeded)
        return self._own_ready() | seeded

    def active_thresholds(self) -> Tensor:
        """Return the thresholds of ready labels and ``-inf`` elsewhere.

        Returns
        -------
        Tensor, shape (C, K)
            Thresholds in logit space; ``-inf`` disables boosting for labels not yet ready.
        """
        thresholds = self.thresholds
        return torch.where(
            self.ready.unsqueeze(1), thresholds, torch.full_like(thresholds, -math.inf)
        )

    # -------------------------------------------------------------------------------- updates

    def _follow(self, device: torch.device) -> None:
        estimate: Tensor = self.estimate
        if estimate.device != device:
            self.to(device)

    def _negatives(
        self, logits: Tensor, targets: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Return detached float32 logits and the boolean negative mask, both ``(N, C)``."""
        z, y, m = as_label_matrix(logits, targets, mask)
        if z.shape[1] != self._num_labels:
            raise ValueError(
                f"The tracker has num_labels={self._num_labels} but the logits have "
                f"{z.shape[1]} labels; build it with num_labels={z.shape[1]}."
            )
        self._follow(z.device)
        values = z.detach().float()
        negative = (y.detach() < 0.5) & torch.isfinite(values)
        if m is not None:
            negative = negative & (m.detach() != 0)
        if self.sync_distributed and is_distributed():
            packed = all_gather_rows(torch.where(negative, values, torch.nan))
            negative = ~torch.isnan(packed)
            values = torch.nan_to_num(packed, nan=0.0)
        return values, negative

    def update(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> None:
        """Feed a batch of logits into the estimate.

        Masked elements, positives, targets of exactly 0.5 and non-finite logits are skipped.
        No host synchronisation happens on a single process; under DDP the negatives are
        all-gathered (a collective: call it on every rank).

        Parameters
        ----------
        logits : Tensor, shape (N,), (N, 1) or (N, C)
            Model outputs; detached internally.
        targets : Tensor, same shape as ``logits``
            Targets; elements with target ``< 0.5`` are negatives and feed the estimate.
        mask : Tensor, same shape as ``logits``, optional
            Non-zero elements are used.
        """
        if self._strategy == "fixed":
            as_label_matrix(logits, targets, mask)
            return
        values, negative = self._negatives(logits, targets, mask)
        with torch.no_grad():
            if self._strategy in ("batch", "ema"):
                self._update_moments(values, negative)
            elif self._strategy == "window":
                self._update_window(values, negative)
            else:
                self._update_reservoir(values, negative)
        self._num_updates += 1
        if self._num_updates == _READY_CHECK_AT:
            self._check_ready(f"after {_READY_CHECK_AT} updates")

    def _batch_quantile(self, values: Tensor, negative: Tensor) -> tuple[Tensor, Tensor]:
        padded = torch.where(negative, values, torch.full_like(values, math.inf)).T
        count = negative.sum(0)
        return _sorted_quantile(padded, count, self._levels), count

    def _update_moments(self, values: Tensor, negative: Tensor) -> None:
        q, count = self._batch_quantile(values, negative)
        has = (count > 0).unsqueeze(1)
        estimate: Tensor = self.estimate
        total: Tensor = self.total_seen
        support: Tensor = self.support
        if self._strategy == "batch":
            estimate.copy_(torch.where(has, q, estimate.float()))
            support.copy_(torch.where(count > 0, count, support))
        else:
            first = has & (total == 0).unsqueeze(1)
            m = self._momentum
            blended = m * estimate.float() + (1.0 - m) * q
            estimate.copy_(torch.where(first, q, torch.where(has, blended, estimate.float())))
            support.copy_(total + count)
        total.add_(count)

    @staticmethod
    def _ranks(negative: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(C, N)`` negatives and each one's rank among this batch's negatives.

        The scan runs along long contiguous rows: a cumsum over the outer dimension of an
        ``(N, C)`` tensor is very slow on GPU.
        """
        neg = negative.T.contiguous()
        return neg, torch.cumsum(neg.to(torch.int32), dim=1).long() - 1

    def _update_window(self, values: Tensor, negative: Tensor) -> None:
        cap = self._capacity_value
        buffer: Tensor = self.buffer
        total: Tensor = self.total_seen
        count: Tensor = self.buffer_count
        neg, rank = self._ranks(negative)  # (C, N)
        vals = values.T
        new = neg.sum(1)  # (C,)
        keep = neg & (rank >= (new - cap).unsqueeze(1))  # at most the last `cap`
        slot = (total.unsqueeze(1) + rank) % cap
        slot = torch.where(keep, slot, torch.full_like(slot, cap))  # column `cap` = discard
        source = torch.where(keep, vals, torch.full_like(vals, math.inf))  # discard stays +inf
        buffer.scatter_(1, slot, source.to(buffer.dtype))
        total.add_(new)
        count.copy_(torch.clamp(count + new, max=cap))
        self.estimate.copy_(_sorted_quantile(buffer[:, :cap], count, self._levels))
        self.support.copy_(count)

    def _update_reservoir(self, values: Tensor, negative: Tensor) -> None:
        cap = self._capacity_value
        buffer: Tensor = self.buffer
        total: Tensor = self.total_seen
        count: Tensor = self.buffer_count
        c = self._num_labels
        neg, rank = self._ranks(negative)  # (C, N)
        vals = values.T
        index = total.unsqueeze(1) + rank  # position of each negative in the epoch stream
        label = torch.arange(c, device=values.device).unsqueeze(1)
        key = _hash32(_hash32(index) ^ _hash32(label * 0x9E3779B1 + self._seed))
        draw = (key * (index + 1)) >> 32  # uniform in [0, index]
        slot = torch.where(index < cap, index, torch.where(draw < cap, draw, cap))
        slot = torch.where(neg, slot, torch.full_like(slot, cap))
        # Algorithm R is sequential: within a batch the latest negative wins a slot.
        winner = torch.full((c, cap + 1), -1, dtype=torch.int64, device=values.device)
        winner.scatter_reduce_(1, slot, index, reduce="amax")
        wins = neg & (winner.gather(1, slot) == index) & slot.lt(cap)
        slot = torch.where(wins, slot, torch.full_like(slot, cap))
        source = torch.where(wins, vals, torch.full_like(vals, math.inf))
        buffer.scatter_(1, slot, source.to(buffer.dtype))
        total.add_(neg.sum(1))
        count.copy_(torch.clamp(total, max=cap))

    def calibrate(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> None:
        """Set the thresholds from scores (e.g. a validation pass) as initial thresholds.

        Uses :func:`aucreshape.metrics.threshold_at_specificity`, the exact operating point
        on these scores, so the positives boosted afterwards are this set's false negatives.
        For ``fixed`` trackers these become the thresholds; other strategies use them until
        their own estimate is valid. Scores are all-gathered under DDP. Warns when a label
        has fewer than ``min_negatives`` negatives. May synchronise with the host.

        Parameters
        ----------
        logits : Tensor, shape (N,), (N, 1) or (N, C)
            Logits of the reference set.
        targets : Tensor, same shape as ``logits``
            Targets of the reference set.
        mask : Tensor, same shape as ``logits``, optional
            Non-zero elements are used.
        """
        z, y, m = as_label_matrix(logits, targets, mask)
        if z.shape[1] != self._num_labels:
            raise ValueError(
                f"The tracker has num_labels={self._num_labels} but the logits have "
                f"{z.shape[1]} labels; build it with num_labels={z.shape[1]}."
            )
        keep = torch.ones_like(z, dtype=torch.bool) if m is None else m != 0
        z = z.detach().float()
        y = y.detach().float()
        if self.sync_distributed and is_distributed():
            packed = all_gather_rows(torch.cat([z, y, keep.float()], dim=1))
            c = self._num_labels
            z, y, keep = packed[:, :c], packed[:, c : 2 * c], packed[:, 2 * c :] != 0
        negatives = ((y < 0.5) & keep).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # missing classes handled below
            thr = metrics.threshold_at_specificity(z, y, list(self._specs), mask=keep)
        thr = thr.reshape(self._num_labels, len(self._specs)).float()
        thr = torch.nan_to_num(thr, nan=-math.inf)
        few = (negatives < self._min_negatives).tolist()
        if any(few):
            labels = [i for i, f in enumerate(few) if f]
            warnings.warn(
                f"ThresholdTracker.calibrate: label(s) {labels} have fewer than "
                f"min_negatives={self._min_negatives} negatives "
                f"({negatives[labels].tolist()}); thresholds at specificity "
                f"{max(self._specs)} from so few samples are unreliable.",
                UserWarning,
                stacklevel=2,
            )
        initial: Tensor = self.initial
        initial.copy_(thr.to(initial.device))
        self.seeded.fill_(True)

    def end_epoch(self) -> None:
        """Finalise the ``epoch`` strategy and warn once if a label is still not ready.

        For ``epoch``, the reservoir's quantiles become the estimate of every label that saw
        negatives this epoch, and the reservoir is cleared; calling it twice is harmless.
        May synchronise with the host (readiness check); it is not a collective.
        """
        if self._strategy == "epoch":
            with torch.no_grad():
                buffer: Tensor = self.buffer
                count: Tensor = self.buffer_count
                cap = self._capacity_value
                has = count > 0
                q = _sorted_quantile(buffer[:, :cap], count, self._levels)
                estimate: Tensor = self.estimate
                support: Tensor = self.support
                estimate.copy_(torch.where(has.unsqueeze(1), q, estimate.float()))
                support.copy_(torch.where(has, count, support))
                buffer.fill_(math.inf)
                count.zero_()
                self.total_seen.zero_()
        self._check_ready("at the end of the epoch")

    def _check_ready(self, when: str) -> None:
        if self._warned or self._strategy == "fixed":
            return
        ready = self.ready.tolist()
        if all(ready):
            return
        self._warned = True
        labels = [i for i, r in enumerate(ready) if not r]
        support = self.support[labels].tolist()
        hint = {
            "batch": "batches need that many negatives each; prefer strategy='window'",
            "epoch": "an epoch needs that many negatives",
        }.get(self._strategy, "the tracker needs that many negatives")
        warnings.warn(
            f"ThresholdTracker is not ready {when} for label(s) {labels}: it has "
            f"{support} of the min_negatives={self._min_negatives} negatives needed for "
            f"target_specificity={max(self._specs)} ({hint}). Boosting is off for these "
            "labels. Fixes: a larger window_size, strategy='epoch', thresholds from a "
            "validation pass (ThresholdTracker.from_scores), or a lower target_specificity.",
            UserWarning,
            stacklevel=3,
        )

    def reset(self) -> None:
        """Forget all accumulated statistics (initial thresholds are kept).

        Warnings are re-armed. ``fixed`` trackers keep their thresholds.
        """
        with torch.no_grad():
            self.estimate.fill_(-math.inf)
            self.support.zero_()
            if self._strategy != "fixed":
                self.total_seen.zero_()
            if self._capacity_value:
                self.buffer.fill_(math.inf)
                self.buffer_count.zero_()
        self._num_updates = 0
        self._warned = False

    # ------------------------------------------------------------------------ state handling

    def _config(self) -> dict[str, Any]:
        return {
            "target_specificity": list(self._specs),
            "num_labels": self._num_labels,
            "strategy": self._strategy,
            "capacity": self._capacity_value,
            "min_negatives": self._min_negatives,
            "seed": self._seed,
        }

    def get_extra_state(self) -> dict[str, Any]:
        """Return the configuration, stored in ``state_dict`` and checked on load."""
        return self._config()

    def set_extra_state(self, state: Any) -> None:  # noqa: ANN401 - signature of nn.Module
        """Check that a loaded state was saved by an identically configured tracker."""
        mine = self._config()
        diffs = [
            f"{k}: saved {state.get(k)!r}, here {v!r}" for k, v in mine.items() if state.get(k) != v
        ]
        if diffs:
            raise ValueError(
                "Cannot load a ThresholdTracker state saved with a different configuration ("
                + "; ".join(diffs)
                + "). Build the tracker with the saved configuration first."
            )

    def extra_repr(self) -> str:
        """Summarise the configuration for ``print(tracker)``."""
        specs = ", ".join(f"{s:g}" for s in self._specs)
        extra = f", capacity={self._capacity_value}" if self._capacity_value else ""
        return (
            f"target_specificity=({specs}), num_labels={self._num_labels}, "
            f"strategy={self._strategy!r}, min_negatives={self._min_negatives}{extra}"
        )

    @classmethod
    def from_scores(
        cls,
        logits: Tensor,
        targets: Tensor,
        target_specificity: float | Sequence[float],
        *,
        mask: Tensor | None = None,
        sync_distributed: bool = True,
    ) -> ThresholdTracker:
        """Build a ``fixed`` tracker from scores, e.g. from a validation pass.

        Parameters
        ----------
        logits : Tensor, shape (N,), (N, 1) or (N, C)
            Logits of the reference set.
        targets : Tensor, same shape as ``logits``
            Targets of the reference set.
        target_specificity : float or sequence of float
            Target specificities, strictly ascending.
        mask : Tensor, same shape as ``logits``, optional
            Non-zero elements are used.
        sync_distributed : bool, default True
            All-gather the scores across ranks when ``torch.distributed`` is initialised.

        Returns
        -------
        ThresholdTracker
            A tracker with ``strategy="fixed"`` whose thresholds are the operating points of
            the given scores (see :meth:`calibrate`).

        Examples
        --------
        >>> import torch
        >>> from aucreshape import ThresholdTracker
        >>> logits = torch.arange(100.0)  # 100 validation negatives
        >>> tracker = ThresholdTracker.from_scores(logits, torch.zeros(100), 0.9)
        >>> tracker.thresholds  # predicting positive iff logit >= 90 flags 10% of negatives
        tensor([[90.]])
        """
        z, _, _ = as_label_matrix(logits, targets, mask)
        specs = check_specificities(target_specificity)
        tracker = cls(
            specs,
            num_labels=z.shape[1],
            strategy="fixed",
            initial_thresholds=torch.full((z.shape[1], len(specs)), -math.inf),
            sync_distributed=sync_distributed,
        )
        tracker.calibrate(logits, targets, mask)
        return tracker.to(logits.device)
