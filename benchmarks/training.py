"""Pretrain -> fine-tune -> evaluate, with an identical protocol for every method.

Every method fine-tunes from the same pretrained checkpoint per seed, with the same optimiser,
learning rate, epochs and batch size; only the loss differs. The epoch is selected on the
validation sensitivity at the target specificity.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from aucreshape import AUCReshapingLoss, auc_reshaping_loss, boost_values, metrics
from benchmarks.data import Split

LossFn = Callable[[Tensor, Tensor], Tensor]


class MLP(nn.Sequential):
    def __init__(
        self,
        num_features: int,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        num_outputs: int = 1,
    ) -> None:
        layers: list[nn.Module] = []
        width = num_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, num_outputs))
        super().__init__(*layers)


class ResNet18(nn.Module):
    """ImageNet-pretrained ResNet-18 for 1-channel images in [0, 1]."""

    def __init__(self, num_outputs: int) -> None:
        from torchvision.models import ResNet18_Weights, resnet18

        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_outputs)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x.expand(-1, 3, -1, -1)
        return self.net((x - self.mean) / self.std)  # type: ignore[no-any-return]


@dataclass
class TrainConfig:
    model: str = "mlp"  # or "resnet18"
    augment: bool = False  # random horizontal flips (images)
    hidden: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    optimizer: str = "adam"  # or "sgd"
    pretrain_lr: float = 1e-3
    pretrain_epochs: int = 8
    finetune_lr: float = 3e-4
    finetune_epochs: int = 6
    batch_size: int = 256
    pos_weight: float | None = None
    select_specificity: float = 0.98
    device: str = "cpu"


@dataclass
class Method:
    """A fine-tuning loss.

    ``kind`` is "bce", "aucreshape" (kwargs -> AUCReshapingLoss) or "custom" (``build`` returns
    the loss; a loss with ``needs_index = True`` is called as ``loss(logits, y, index)``).
    """

    name: str
    kind: str = "bce"
    kwargs: dict[str, Any] = field(default_factory=dict)
    calibrate: bool = False  # warm-start the tracker from the pretrained model's train scores
    build: Callable[[TrainConfig, Split], Any] | None = None


def make_model(split: Split, cfg: TrainConfig) -> nn.Module:
    out = 1 if split.y_train.dim() == 1 else split.y_train.shape[1]
    if cfg.model == "resnet18":
        return ResNet18(out)
    return MLP(split.num_features, cfg.hidden, cfg.dropout, out)


def _optimizer(model: nn.Module, cfg: TrainConfig, lr: float) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=lr)


@torch.no_grad()
def predict(model: nn.Module, x: Tensor, cfg: TrainConfig, batch: int = 1024) -> Tensor:
    model.eval()
    outs = [model(x[i : i + batch].to(cfg.device)).cpu() for i in range(0, len(x), batch)]
    out = torch.cat(outs)
    return out.squeeze(-1) if out.shape[-1] == 1 else out


def _epoch(
    model: nn.Module,
    split: Split,
    loss_fn: LossFn,
    opt: torch.optim.Optimizer,
    cfg: TrainConfig,
    gen: torch.Generator,
) -> float:
    model.train()
    order = torch.randperm(len(split.x_train), generator=gen)
    total = 0.0
    for i in range(0, len(order), cfg.batch_size):
        idx = order[i : i + cfg.batch_size]
        x = split.x_train[idx].to(cfg.device)
        y = split.y_train[idx].to(cfg.device)
        if cfg.augment:
            flip = torch.rand(len(x), generator=gen) < 0.5
            x = torch.where(flip.to(cfg.device).view(-1, 1, 1, 1), x.flip(-1), x)
        logits = model(x)
        logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits
        if getattr(loss_fn, "needs_index", False):
            loss = loss_fn(logits, y, idx.to(cfg.device))
        else:
            loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.detach())
    return total


def bce(cfg: TrainConfig) -> LossFn:
    pw = None if cfg.pos_weight is None else torch.tensor(cfg.pos_weight, device=cfg.device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def pretrain(split: Split, seed: int, cfg: TrainConfig) -> dict[str, Tensor]:
    torch.manual_seed(seed)
    model = make_model(split, cfg).to(cfg.device)
    opt = _optimizer(model, cfg, cfg.pretrain_lr)
    gen = torch.Generator().manual_seed(seed)
    loss_fn = bce(cfg)
    for _ in range(cfg.pretrain_epochs):
        _epoch(model, split, loss_fn, opt, cfg, gen)
    return copy.deepcopy(model.state_dict())


def make_loss(method: Method, cfg: TrainConfig, split: Split | None = None) -> LossFn:
    if method.kind == "custom":
        assert method.build is not None
        assert split is not None
        return method.build(cfg, split)  # type: ignore[no-any-return]
    if method.kind == "bce":
        return bce(cfg)
    if method.kind == "aucreshape":
        kwargs = dict(method.kwargs)
        if cfg.pos_weight is not None:
            kwargs.setdefault("pos_weight", cfg.pos_weight)
        return AUCReshapingLoss(**kwargs).to(cfg.device)
    raise ValueError(f"unknown method kind {method.kind!r}")


@torch.no_grad()
def _diagnostics(
    model: nn.Module, split: Split, loss_fn: AUCReshapingLoss, cfg: TrainConfig
) -> dict[str, Any]:
    """Thresholds, boosted fraction and boosted share of the loss on the whole training set."""
    logits = predict(model, split.x_train, cfg)
    y = split.y_train
    thresholds = loss_fn.tracker.active_thresholds().cpu()
    boost = loss_fn._boost()
    b = boost_values(logits, y, thresholds, boost, combine=loss_fn.combine)
    per = auc_reshaping_loss(
        logits,
        y,
        thresholds,
        boost,
        mode=loss_fn.mode,
        combine=loss_fn.combine,
        reduction="none",
        pos_weight=cfg.pos_weight,
        min_ratio=loss_fn.min_ratio,
    )
    positive = y > 0.5
    return {
        "thresholds": loss_fn.thresholds.cpu().tolist(),
        "ready": loss_fn.tracker.ready.cpu().tolist(),
        "boosted_fraction": float((b[positive] > 0).float().mean()),
        "mean_boost": float(b[b > 0].mean()) if (b > 0).any() else 0.0,
        "boosted_loss_share": float(per[b > 0].sum() / per.sum()),
    }


def finetune(
    split: Split, seed: int, state: dict[str, Tensor], method: Method, cfg: TrainConfig
) -> dict[str, Any]:
    """Fine-tune from ``state``; return the selected epoch's logits and per-epoch history."""
    torch.manual_seed(seed + 1)
    model = make_model(split, cfg).to(cfg.device)
    model.load_state_dict(state)
    opt = _optimizer(model, cfg, cfg.finetune_lr)
    gen = torch.Generator().manual_seed(seed + 1)
    loss_fn = make_loss(method, cfg, split)
    if method.calibrate and isinstance(loss_fn, AUCReshapingLoss):
        loss_fn.tracker.calibrate(predict(model, split.x_train, cfg), split.y_train)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    start = time.perf_counter()
    for epoch in range(cfg.finetune_epochs):
        train_loss = _epoch(model, split, loss_fn, opt, cfg, gen)
        if isinstance(loss_fn, AUCReshapingLoss):
            loss_fn.end_epoch()
        val_logits = predict(model, split.x_val, cfg)
        score = float(
            metrics.sensitivity_at_specificity(val_logits, split.y_val, cfg.select_specificity)
            .float()
            .mean()
        )
        record: dict[str, Any] = {"epoch": epoch + 1, "train_loss": train_loss, "val_select": score}
        if isinstance(loss_fn, AUCReshapingLoss):
            record.update(_diagnostics(model, split, loss_fn, cfg))
        history.append(record)
        if best is None or score > best["val_select"]:
            best = {
                "epoch": epoch + 1,
                "val_select": score,
                "val_logits": val_logits,
                "test_logits": predict(model, split.x_test, cfg),
            }
    assert best is not None
    best["history"] = history
    best["train_seconds"] = time.perf_counter() - start
    return best
