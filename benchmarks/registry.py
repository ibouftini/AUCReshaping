"""Dataset and method registries for the config-driven benchmark runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from benchmarks import data
from benchmarks.training import Method, TrainConfig


def _imblearn(name: str, seed: int = 0) -> data.Split:
    from imblearn.datasets import fetch_datasets

    bunch = fetch_datasets(data_home=str(data.CACHE / "imblearn"), filter_data=(name,))[name]
    x = np.asarray(bunch.data, dtype=np.float32)
    y = (np.asarray(bunch.target) == 1).astype(np.float32)
    idx = data._stratified_split(x, y, (0.6, 0.2, 0.2), seed)
    return data._to_split(name, x, y, idx, note="imbalanced-learn benchmark set")


DATASETS: dict[str, Callable[..., data.Split]] = {
    "synthetic": data.synthetic,
    "creditcard": data.creditcard,
    "mammography": lambda seed=0: _imblearn("mammography", seed),
    "protein_homo": lambda seed=0: _imblearn("protein_homo", seed),
    "breastmnist": lambda **kw: data.medmnist("breastmnist", **kw),
    "pneumoniamnist": lambda **kw: data.medmnist("pneumoniamnist", **kw),
}


class FocalLoss:
    """Sigmoid focal loss (torchvision), mean-reduced, alpha disabled."""

    def __init__(self, gamma: float) -> None:
        from torchvision.ops import sigmoid_focal_loss

        self.fn = sigmoid_focal_loss
        self.gamma = gamma

    def __call__(self, logits: Tensor, y: Tensor) -> Tensor:
        return self.fn(logits, y, alpha=-1, gamma=self.gamma, reduction="mean")


class LibAUCPartialAUC:
    """LibAUC one-way partial-AUC loss (``pAUC_DRO_Loss``) on sigmoid scores."""

    needs_index = True

    def __init__(self, data_len: int, device: str, **kwargs: float) -> None:
        from libauc.losses import pAUC_DRO_Loss

        self.loss = pAUC_DRO_Loss(data_len=data_len, device=device, **kwargs)

    def __call__(self, logits: Tensor, y: Tensor, index: Tensor) -> Tensor:
        # The loss is undefined without both classes (LibAUC expects an oversampling
        # DualSampler); under the shared-batches protocol such batches are skipped.
        if not (bool((y > 0.5).any()) and bool((y < 0.5).any())):
            return (logits * 0.0).sum()
        return self.loss(torch.sigmoid(logits), y, index)


def make_method(name: str, params: dict[str, Any], target: float) -> Method:
    """Build a benchmark method from its registry name and one grid configuration."""
    label = name + ("" if not params else " " + ",".join(f"{k}={v}" for k, v in params.items()))
    if name == "bce":
        return Method(label)
    if name == "weighted_bce":

        def build(cfg: TrainConfig, split: data.Split) -> Any:
            base = cfg.pos_weight or 1.0
            pw = torch.tensor(base * params["pos_weight"], device=cfg.device)
            return torch.nn.BCEWithLogitsLoss(pos_weight=pw)

        return Method(label, "custom", build=build)
    if name == "focal":
        return Method(label, "custom", build=lambda cfg, split: FocalLoss(params["gamma"]))
    if name == "libauc_pauc":
        return Method(
            label,
            "custom",
            build=lambda cfg, split: LibAUCPartialAUC(len(split.x_train), cfg.device, **params),
        )
    if name.startswith("aucreshape"):
        mode = name.split("_", 1)[1] if "_" in name else "prob"
        kwargs = {"target_specificity": target, "mode": mode, **params}
        return Method(label, "aucreshape", kwargs)
    raise ValueError(f"unknown method {name!r}")
