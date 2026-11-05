"""Training losses of the evaluated arms.

AUCReshaping arms use ``AUCReshapingLoss`` itself, since that is what users run. The
controls need a per-sample boost assignment the public API does not expose, so
``ControlLoss`` recomputes the loss from ``boost_values`` and a ``ThresholdTracker``; with the
``"threshold"`` rule it reproduces ``AUCReshapingLoss`` exactly (tested), so any difference
between AUCReshaping and a control comes from the rule alone:

- ``random``: the same boosts, moved to randomly chosen positives (isolates the selection);
- ``all``: every positive boosted once the tracker is ready (targeting vs re-weighting);
- ``above``: only positives above the highest threshold (placebo).

``heldout`` and ``oracle`` are ``AUCReshapingLoss`` with a ``fixed`` tracker that the runner
recalibrates every epoch on validation scores, or on a large fresh sample (synthetic only).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from aucreshape import AUCReshapingLoss, ThresholdTracker, boost_values
from benchmarks.registry import FocalLoss, LibAUCPartialAUC

Loss = Callable[..., Tensor]


def boosted_bce(
    logits: Tensor,
    targets: Tensor,
    boost: Tensor,
    mode: str = "prob",
    pos_weight: float | None = None,
    min_ratio: float = 0.1,
) -> Tensor:
    """Mean binary cross-entropy with a given per-sample boost (0 for unboosted samples)."""
    z, y, b = logits.float(), targets.float(), boost.float()
    w = 1.0 if pos_weight is None else pos_weight
    log_p = F.logsigmoid(z)
    if mode == "prob":
        # log(p - b') with b' = min(b, (1 - r) sg(p)): rho = b'/sg(p) carries no gradient, and
        # exp(sg(log p) - log p) is 1 in value with the gradient of sg(p)/p.
        rho = torch.where(
            b > 0, torch.clamp(b * (1 + torch.exp(-z.detach())), max=1 - min_ratio), 0.0
        )
        positive = log_p + torch.log1p(-rho * torch.exp(log_p.detach() - log_p))
    elif mode == "logit":
        positive = F.logsigmoid(z - b)
    elif mode == "weight":
        positive = (1 + b) * log_p
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return -(w * y * positive + (1 - y) * F.logsigmoid(-z)).mean()


class ControlLoss(nn.Module):
    """AUCReshaping with a different rule for which positives receive the boosts."""

    def __init__(
        self,
        rule: str,
        target_specificity: float | Sequence[float],
        *,
        boost: float | Sequence[float],
        mode: str = "prob",
        combine: str = "band",
        pos_weight: float | None = None,
        min_ratio: float = 0.1,
        seed: int = 0,
        **tracker_kwargs: Any,
    ) -> None:
        super().__init__()
        if rule not in {"threshold", "random", "all", "above", "negatives"}:
            raise ValueError(f"unknown rule {rule!r}")
        self.rule, self.mode, self.combine = rule, mode, combine
        self.boost = boost
        self.pos_weight, self.min_ratio = pos_weight, min_ratio
        self.tracker = ThresholdTracker(target_specificity, **tracker_kwargs)
        self.gen = torch.Generator().manual_seed(seed)
        self.last_boosted = torch.zeros(0, dtype=torch.bool)
        self.last_eligible = torch.zeros(0, dtype=torch.bool)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if not (self.training and torch.is_grad_enabled()):
            pw = None if self.pos_weight is None else torch.tensor(self.pos_weight)
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        z = logits.detach()
        positive = targets > 0.5
        active = self.tracker.active_thresholds()
        if self.rule == "negatives":  # the false positives at the operating point, as weights
            top = active[..., -1].reshape(())
            self.last_boosted = (~positive) & (z >= top) & torch.isfinite(top)
            self.last_eligible = ~positive
            weight = 1 + float(self.boost) * self.last_boosted.float()  # type: ignore[arg-type]
            w = 1.0 if self.pos_weight is None else self.pos_weight
            loss = -(
                w * targets * F.logsigmoid(logits) + weight * (1 - targets) * F.logsigmoid(-logits)
            )
            self.tracker.update(z, targets)
            return loss.mean()
        if self.rule in {"all", "above"}:
            ready = torch.isfinite(active)
            everyone = torch.where(ready, math.inf, -math.inf)
            b = boost_values(z, targets, everyone, self.boost, combine=self.combine)
            if self.rule == "above":
                b = torch.where(z.reshape(-1, 1) >= active[..., -1], b.reshape(-1, 1), 0.0)
        else:
            b = boost_values(z, targets, active, self.boost, combine=self.combine)
            if self.rule == "random":
                b = b.reshape(-1)
                pos = torch.nonzero(positive.reshape(-1), as_tuple=True)[0]
                perm = torch.randperm(len(pos), generator=self.gen).to(pos.device)
                moved = torch.zeros_like(b)
                moved[pos[perm]] = b[pos]
                b = moved
        b = b.reshape(logits.shape)
        self.last_boosted = (b > 0) & positive
        self.last_eligible = positive
        loss = boosted_bce(logits, targets, b, self.mode, self.pos_weight, self.min_ratio)
        self.tracker.update(z, targets)
        return loss

    def get_extra_state(self) -> dict[str, Tensor]:
        return {"gen": self.gen.get_state()}

    def set_extra_state(self, state: dict[str, Tensor]) -> None:
        self.gen.set_state(state["gen"])


class SmoothedLoss(nn.Module):
    """AUCReshaping with a per-sample moving average of the boost (I5, maintainer's proposal).

    Each training sample keeps b_i <- m b_i + (1 - m) b_i(now), updated whenever it is in a
    batch, so a positive that crosses the threshold back and forth gets a steady weight
    instead of an on/off one. Needs the samples' indices (``needs_index``).
    """

    needs_index = True

    def __init__(
        self,
        target_specificity: float | Sequence[float],
        *,
        boost: float | Sequence[float],
        momentum: float,
        data_len: int,
        mode: str = "prob",
        combine: str = "band",
        pos_weight: float | None = None,
        min_ratio: float = 0.1,
        **tracker_kwargs: Any,
    ) -> None:
        super().__init__()
        self.boost, self.momentum, self.mode, self.combine = boost, momentum, mode, combine
        self.pos_weight, self.min_ratio = pos_weight, min_ratio
        self.tracker = ThresholdTracker(target_specificity, **tracker_kwargs)
        self.register_buffer("ema", torch.zeros(data_len))
        self.last_boosted = torch.zeros(0, dtype=torch.bool)
        self.last_eligible = torch.zeros(0, dtype=torch.bool)

    def forward(self, logits: Tensor, targets: Tensor, index: Tensor) -> Tensor:
        if not (self.training and torch.is_grad_enabled()):
            pw = None if self.pos_weight is None else torch.tensor(self.pos_weight)
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        if self.ema.device != logits.device:
            self.to(logits.device)
        z = logits.detach()
        now = boost_values(z, targets, self.tracker.active_thresholds(), self.boost,
                           combine=self.combine).reshape(-1)  # fmt: skip
        with torch.no_grad():
            self.ema[index] = self.momentum * self.ema[index] + (1 - self.momentum) * now
        b = self.ema[index].reshape(logits.shape)
        self.last_boosted = (b > 0) & (targets > 0.5)
        self.last_eligible = targets > 0.5
        loss = boosted_bce(logits, targets, b, self.mode, self.pos_weight, self.min_ratio)
        self.tracker.update(z, targets)
        return loss


class PlainBCE(nn.Module):
    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def make_arm(
    name: str,
    params: dict[str, Any],
    target: float,
    pos_weight: float | None,
    seed: int,
    data_len: int = 0,
    device: str = "cpu",
) -> Loss:
    """Build the training loss of one arm; ``params`` is one configuration from its grid."""
    params = dict(params)
    if name == "ce":
        return PlainBCE(pos_weight)
    if name == "weighted_bce":
        return PlainBCE((pos_weight or 1.0) * params["pos_weight"])
    if name == "focal":
        return FocalLoss(params["gamma"])
    if name == "libauc_pauc":
        return LibAUCPartialAUC(data_len, device, **params)
    specs = params.pop("target_specificity", target)
    if name in {"aucreshape", "warm_aucreshape"}:  # warm: params carry ``warmup_steps``
        return AUCReshapingLoss(specs, pos_weight=pos_weight, **params)
    if name == "aucreshape_pw":
        return AUCReshapingLoss(specs, pos_weight=params.pop("pos_weight"), **params)
    if name == "near_miss":  # boost only between 8 specificity points below the target and it
        b = params.pop("boost")
        return AUCReshapingLoss([round(specs - 0.08, 6), specs], boost=[0.0, b], combine="band",
                                pos_weight=pos_weight, **params)  # fmt: skip
    if name == "hard_negatives":
        return ControlLoss("negatives", specs, mode="weight", pos_weight=pos_weight, **params)
    if name == "smoothed":
        return SmoothedLoss(specs, pos_weight=pos_weight, data_len=data_len, **params)
    if name in {"heldout", "oracle"}:
        k = 1 if isinstance(specs, float) else len(specs)
        tracker = ThresholdTracker(specs, strategy="fixed", initial_thresholds=[math.inf] * k)
        loss = AUCReshapingLoss(pos_weight=pos_weight, tracker=tracker, **params)
        loss.threshold_source = "val" if name == "heldout" else "oracle"  # type: ignore[assignment]
        return loss
    rules = {
        "threshold_control": "threshold",
        "random_boost": "random",
        "boost_all": "all",
        "placebo": "above",
    }
    if name in rules:
        return ControlLoss(rules[name], specs, pos_weight=pos_weight, seed=seed + 7919, **params)
    raise ValueError(f"unknown arm {name!r}")


def calibrate(loss: Loss, logits: Tensor, targets: Tensor) -> None:
    """Set a ``heldout``/``oracle`` arm's thresholds from scores (validation or oracle)."""
    loss.tracker.calibrate(logits, targets)  # type: ignore[attr-defined]


def boosted_fraction(loss: Loss) -> float | None:
    """Fraction of the last batch's positives that were boosted, for reshaping arms."""
    if isinstance(loss, AUCReshapingLoss):
        return float(loss.last_stats["boosted_fraction"].mean())
    if isinstance(loss, ControlLoss | SmoothedLoss):
        return float(loss.last_boosted.float().sum() / max(1, int(loss.last_eligible.sum())))
    return None
