"""A replication of the paper's protocol. AUCReshaping replaces cross-entropy for the whole
supervised stage, starting from the same pretrained weights as the baseline (Bhat et al. 2023,
p. 3 and p. 8).

Jobs:
- ``nih``: NIH ChestX-ray14 from the preprocessing step (``nih256`` kernel output),
  box-averaged to 128 px, official patient-wise test split, 10% of the remaining patients for
  validation. Same 15-output model as ``chestmnist``; 200 epochs, evaluated every epoch.
- ``chestmnist``: MedMNIST+ ChestMNIST (NIH ChestX-ray14) at 64/128/224 px. One 15-output
  model (14 findings + "any finding"); the paper's tasks are pleural effusion, pneumothorax
  and normal-vs-abnormal. Backbone: SwAV ResNet-50 (the paper's) or ImageNet ResNet-18/50.
- ``pneumoniamnist``: long training (epoch scaling), ImageNet ResNet backbone.
- ``creditcard``: the paper's MLP, SGD, batch 100, pos_weight 5, 200 epochs from scratch.

Paper convention throughout: normal = positive (label 1), and specificity is measured on the
abnormal class. The model predicts abnormality logits ``z``; the loss sees ``(-z, 1 - y)``.
Both arms use ``AUCReshapingLoss``; the baseline has ``boost=0``, which equals weighted BCE
exactly. Epochs are selected on the lowest validation loss (paper) and, as a second view, on
validation sensitivity at 0.98. Finished runs (rows in ``OUT/runs.jsonl``) are skipped; each
row carries ``key``, ``status`` and ``git_sha`` for ``benchmarks.kaggle.collect``, and
``OUT/scores/<key>.npz`` keeps float16 logits at ``CHECKPOINTS`` and at the selected epochs.
A run past ``AUCRESHAPE_DEADLINE`` (set by the Kaggle kernels) stops after its epoch and keeps
its partial record.

    python -m benchmarks.replication chestmnist --size 128 --arms ce,ar_paper --seeds 0 1 2
    python -m benchmarks.replication nih --data /kaggle/input --arms focal --seeds 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

import aucreshape
from aucreshape import AUCReshapingLoss, boost_values, metrics
from benchmarks.data import CACHE
from benchmarks.review.arms import boosted_bce
from benchmarks.review.nih import FINDINGS

ZENODO = "https://zenodo.org/records/10519652/files/{name}.npz?download=1"
MD5 = {  # MedMNIST+ v3 (medmnist.INFO)
    "chestmnist_64": "9de6cd0b934ebb5b7426cfba5efbae16",
    "chestmnist_128": "db107e5590b27930b62dbaf558aebee3",
    "chestmnist_224": "45bd33e6f06c3e8cdb481c74a89152aa",
    "pneumoniamnist_64": "8f4eceb4ccffa70c672198ea285246c6",
    "pneumoniamnist_128": "05b46931834c231683c68f40c47b2971",
    "pneumoniamnist_224": "d6a3c71de1b945ea11211b03746c1fe1",
}
SWAV_URL = "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"
CHEST_TASKS = {"effusion": 2, "pneumothorax": 7, "abnormal": 14}  # label index in our 15
SPECS = (0.95, 0.98)
CHECKPOINTS = (5, 10, 20, 50, 100, 150, 200)  # epochs whose logits are kept
ARMS: dict[str, dict[str, Any]] = {
    "ce": {"target_specificity": 0.98, "boost": 0.0},
    # Paper's best pleural-effusion configuration (Table 3).
    "ar_paper": {"target_specificity": [0.90, 0.92, 0.95, 0.98], "boost": [0.4, 0.2, 0.1, 0.1]},
    "ar_single": {"target_specificity": 0.98, "boost": 0.2},
    # Paper's credit-card configuration (Table 7): 0.2 at every threshold.
    "ar_paper_cc": {"target_specificity": [0.90, 0.92, 0.95, 0.98, 0.99], "boost": 0.2},
    # class-weighted focal loss, and ar_paper's boosts moved to random
    # positives of the same label.
    "focal": {"focal": 2.0},
    "random_boost": {
        "target_specificity": [0.90, 0.92, 0.95, 0.98],
        "boost": [0.4, 0.2, 0.1, 0.1],
        "control": "random",
    },
}


def download(url: str, path: Path, md5: str | None = None) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # One temporary file per process: parallel workers may fetch the same file.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
        with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f, length=1 << 22)
        tmp.rename(path)
    if md5 is not None:
        digest = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != md5:
            raise RuntimeError(f"{path} does not match its published MD5; delete it and retry.")
    return path


@dataclass
class Data:
    """Images kept as uint8 (N, H, W) to save memory, or float features for tabular data."""

    x: dict[str, Tensor]
    y: dict[str, Tensor]  # float, (N,) or (N, C), disease/abnormal = 1
    tasks: dict[str, int]  # task name -> label column (or 0 for binary)
    image: bool
    patients: dict[str, Tensor] | None = None  # patient id per image (NIH)


def load_medmnist(name: str, size: int) -> Data:
    stem = f"{name}_{size}"
    path = download(ZENODO.format(name=stem), CACHE / "medmnist" / f"{stem}.npz", MD5.get(stem))
    arrays = np.load(path)
    x, y = {}, {}
    for part in ("train", "val", "test"):
        x[part] = torch.from_numpy(np.ascontiguousarray(arrays[f"{part}_images"]))
        labels = torch.from_numpy(arrays[f"{part}_labels"].astype(np.float32))
        if name == "chestmnist":
            labels = torch.cat([labels, labels.amax(1, keepdim=True)], 1)  # + any finding
        else:
            labels = labels[:, 0]  # pneumonia = 1
        y[part] = labels
    tasks = CHEST_TASKS if name == "chestmnist" else {"pneumonia": 0}
    return Data(x, y, tasks, image=True)


def find_nih(root: Path) -> Path:
    """The folder at or below ``root`` that holds the preprocessing output (images, labels)."""
    for path in [root / "labels.npz", *sorted(root.rglob("labels.npz"))]:
        if path.exists() and (path.parent / "images.npy").exists():
            return path.parent
    raise FileNotFoundError(f"no images.npy + labels.npz below {root}")


def load_nih(
    root: Path, size: int = 128, val_fraction: float = 0.1, split_seed: int = 0, chunk: int = 4096
) -> Data:
    """NIH ChestX-ray14 from ``benchmarks.review.nih`` output, box-averaged to ``size`` px.

    The official test list is the test set; ``val_fraction`` of the other patients (drawn with
    ``split_seed``) form the validation set, so no patient is in two parts. Labels are the 14
    findings in NIH order plus "any finding" (15 columns, disease = 1). Raises if an image is
    blank, which would mean the preprocessing failed.
    """
    folder = find_nih(root)
    meta = np.load(folder / "labels.npz", allow_pickle=False)
    columns = [str(c) for c in meta["columns"]]
    findings = meta["labels"][:, [columns.index(f) for f in FINDINGS]].astype(np.float32)
    y = torch.from_numpy(findings)
    y = torch.cat([y, y.amax(1, keepdim=True)], 1)
    images = np.load(folder / "images.npy", mmap_mode="r")
    n, side = images.shape[0], images.shape[1]
    if side % size:
        raise ValueError(f"size {size} does not divide the stored {side} px")
    factor = side // size
    small = torch.empty((n, size, size), dtype=torch.uint8)
    for i in range(0, n, chunk):
        block = torch.from_numpy(np.array(images[i : i + chunk]))  # a writable copy
        blank = torch.nonzero(block.flatten(1).amax(1) == 0)[:, 0]
        if len(blank):
            raise ValueError(f"blank images at rows {(blank[:5] + i).tolist()}")
        pooled = F.avg_pool2d(block.unsqueeze(1).float(), factor).squeeze(1)
        small[i : i + chunk] = pooled.round().to(torch.uint8)
    patient = meta["patient"]
    test = meta["split"] == "test"
    pool = np.unique(patient[~test])
    rng = np.random.default_rng(split_seed)
    held = rng.choice(pool, size=round(val_fraction * len(pool)), replace=False)
    val = np.isin(patient, held) & ~test
    parts = {"train": ~test & ~val, "val": val, "test": test}
    x, labels, patients = {}, {}, {}
    for part, rows in parts.items():
        index = torch.from_numpy(np.flatnonzero(rows))
        x[part], labels[part] = small[index], y[index]
        patients[part] = torch.from_numpy(patient[rows].astype(np.int64))
    tasks = {
        "effusion": FINDINGS.index("Effusion"),
        "pneumothorax": FINDINGS.index("Pneumothorax"),
        "abnormal": len(FINDINGS),
    }
    return Data(x, labels, tasks, image=True, patients=patients)


def load_creditcard() -> Data:
    from benchmarks.data import creditcard

    split = creditcard(seed=0, positive="fraud")
    return Data(
        {"train": split.x_train, "val": split.x_val, "test": split.x_test},
        {"train": split.y_train, "val": split.y_val, "test": split.y_test},
        {"fraud": 0},
        image=False,
    )


class ImageNet(nn.Module):
    """ResNet for 1-channel uint8 images: [0, 255] -> ImageNet-normalised 3 channels."""

    def __init__(self, arch: str, num_outputs: int) -> None:
        super().__init__()
        if arch == "swav_resnet50":
            net = torchvision.models.resnet50()
            ckpt = torch.load(
                download(SWAV_URL, CACHE / "weights" / "swav_800ep_pretrain.pth.tar"),
                map_location="cpu",
                weights_only=False,
            )
            state = ckpt.get("state_dict", ckpt)
            state = {k.removeprefix("module."): v for k, v in state.items()}
            missing, _ = net.load_state_dict(state, strict=False)
            assert set(missing) <= {"fc.weight", "fc.bias"}, missing
        else:
            weights = {"resnet18": "IMAGENET1K_V1", "resnet50": "IMAGENET1K_V2"}[arch]
            net = getattr(torchvision.models, arch)(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_outputs)
        self.net = net.to(memory_format=torch.channels_last)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x: Tensor) -> Tensor:
        x = ((x.float().unsqueeze(1).expand(-1, 3, -1, -1)) - self.mean) / self.std
        return self.net(x.contiguous(memory_format=torch.channels_last))  # type: ignore[no-any-return]


def mlp(num_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(num_features, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, 1),
    )


@torch.no_grad()
def predict(model: nn.Module, x: Tensor, device: str, batch: int = 512) -> Tensor:
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            out.append(model(x[i : i + batch].to(device, non_blocking=True)).float().cpu())
    logits = torch.cat(out)
    return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


def task_metrics(logits: Tensor, y: Tensor, tasks: dict[str, int]) -> dict[str, float]:
    """Paper convention: normal positive, specificity on abnormal -> score = -logit."""
    out: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, col in tasks.items():
            z = logits[:, col] if logits.dim() == 2 else logits
            t = y[:, col] if y.dim() == 2 else y
            for s in SPECS:
                out[f"{name}/sens@{s:g}"] = float(metrics.sensitivity_at_specificity(-z, 1 - t, s))
            out[f"{name}/auc"] = float(metrics.partial_auc(-z, 1 - t))
    return out


@dataclass
class Job:
    name: str
    size: int
    arch: str
    epochs: int
    lr: float
    batch_size: int
    optimizer: str
    pos_weight: Tensor | None
    window_size: int | None
    eval_every: int


class FocalBCE(nn.Module):
    """Class-weighted sigmoid focal loss, mean over elements (the E4 ``focal`` arm).

    Each element's weighted BCE term is scaled by (1 - p_t)^gamma; gamma = 0 is the CE arm.
    Without gradients (validation) it returns plain weighted BCE, as ``AUCReshapingLoss``
    does, so the validation loss means the same thing in every arm.
    """

    pos_weight: Tensor | None

    def __init__(self, gamma: float, pos_weight: Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        z, y = logits.float(), targets.float()
        w = 1.0 if self.pos_weight is None else self.pos_weight
        log_p, log_q = F.logsigmoid(z), F.logsigmoid(-z)
        if self.training and torch.is_grad_enabled():
            positive = torch.exp(self.gamma * log_q) * log_p  # (1 - p)^gamma log p
            negative = torch.exp(self.gamma * log_p) * log_q  # p^gamma log(1 - p)
        else:
            positive, negative = log_p, log_q
        return -(w * y * positive + (1 - y) * negative).mean()

    def end_epoch(self) -> None:
        pass


class RandomBoost(nn.Module):
    """Control: ``base``'s boosts, moved to randomly chosen positives of the same label.

    Uses ``base``'s tracker, boost table, mode and class weights, so the only difference from
    AUCReshaping is which positives receive the boosts; ``shuffle=False`` reproduces ``base``.
    Without gradients it returns ``base``'s validation loss (plain weighted BCE).
    """

    def __init__(self, base: AUCReshapingLoss, seed: int, shuffle: bool = True) -> None:
        super().__init__()
        self.base, self.shuffle = base, shuffle
        self.tracker = base.tracker
        self.gen = torch.Generator().manual_seed(seed + 7919)
        self.last_boosted = torch.zeros(0, dtype=torch.bool)
        self._fraction = torch.zeros(base.tracker.num_labels)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if not (self.training and torch.is_grad_enabled()):
            return self.base(logits, targets)  # type: ignore[no-any-return]
        base = self.base
        z = logits.detach()
        positive = targets > 0.5
        b = boost_values(
            z, targets, self.tracker.active_thresholds(), base.boost_table, combine=base.combine
        )
        b = b.reshape(logits.shape)
        if self.shuffle:
            column = b.unsqueeze(-1) if b.dim() == 1 else b
            moved = torch.zeros_like(column)
            for c in range(column.shape[1]):
                pos = torch.nonzero(positive.reshape(column.shape)[:, c], as_tuple=True)[0]
                perm = torch.randperm(len(pos), generator=self.gen).to(pos.device)
                moved[pos[perm], c] = column[pos, c]
            b = moved.reshape(b.shape)
        pos_weight = base.pos_weight if base.pos_weight is not None else base._pos_weight_value
        loss = boosted_bce(logits, targets, b, base.mode, pos_weight, base.min_ratio)  # type: ignore[arg-type]
        self.last_boosted = (b > 0) & positive
        num_pos = positive.reshape(len(z), -1).sum(0).clamp(min=1).float()
        self._fraction = self.last_boosted.reshape(len(z), -1).sum(0).float() / num_pos
        self.tracker.update(z, targets)
        return loss

    @property
    def last_stats(self) -> dict[str, Tensor]:
        """``boosted_fraction`` (C,) of the last training call, like ``AUCReshapingLoss``."""
        return {"boosted_fraction": self._fraction}

    def end_epoch(self) -> None:
        self.base.end_epoch()


def make_criterion(job: Job, arm: str, data: Data, device: str, seed: int = 0) -> nn.Module:
    cfg = dict(ARMS[arm])
    if "focal" in cfg:
        return FocalBCE(cfg["focal"], job.pos_weight).to(device)
    control = cfg.pop("control", None)
    c = data.y["train"].shape[1] if data.y["train"].dim() == 2 else 1
    specs = cfg["target_specificity"]
    k = len(specs) if isinstance(specs, list) else 1
    boost = cfg["boost"]
    if c > 1:  # boost only the paper's task labels
        row = torch.tensor(boost if isinstance(boost, list) else [boost] * k, dtype=torch.float32)
        table = torch.zeros(c, k)
        for col in data.tasks.values():
            table[col] = row
        boost = table
    loss = AUCReshapingLoss(
        specs, boost=boost, num_labels=c, pos_weight=job.pos_weight, window_size=job.window_size
    )
    if control == "random":
        return RandomBoost(loss, seed).to(device)
    return loss.to(device)


def train_one(job: Job, data: Data, arm: str, seed: int, device: str) -> dict[str, Any]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    c = data.y["train"].shape[1] if data.y["train"].dim() == 2 else 1
    model = (ImageNet(job.arch, c) if data.image else mlp(data.x["train"].shape[1])).to(device)
    if job.optimizer == "sgd":
        opt: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=job.lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=job.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda" and data.image)
    criterion = make_criterion(job, arm, data, device, seed)
    deadline = float(os.environ.get("AUCRESHAPE_DEADLINE", "inf"))
    x_tr, y_tr = data.x["train"], data.y["train"]
    history: list[dict[str, Any]] = []
    best = {"val_loss": (math.inf, None, None), "val_sens": (-math.inf, None, None)}
    saved: dict[str, Tensor] = {}
    status = "ok"
    start = time.perf_counter()
    for epoch in range(1, job.epochs + 1):
        model.train()
        criterion.train()
        order = torch.randperm(len(x_tr), generator=gen)
        total = 0.0
        boosted: Tensor | None = None
        batches = 0
        for i in range(0, len(order), job.batch_size):
            idx = order[i : i + job.batch_size]
            xb = x_tr[idx].to(device, non_blocking=True)
            yb = y_tr[idx].to(device, non_blocking=True)
            if data.image:  # paper augmentations: horizontal flips and intensity scaling
                flip = (torch.rand(len(idx), generator=gen) < 0.5).to(device)
                xb = torch.where(flip.view(-1, 1, 1), xb.flip(-1), xb)
                scale = (0.9 + 0.2 * torch.rand(len(idx), 1, 1, generator=gen)).to(device)
                xb = (xb.float() * scale).clamp(0, 255)
            with torch.autocast(device, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(xb)
            logits = logits.float()
            logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits
            loss = criterion(-logits, 1 - yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            stats = getattr(criterion, "last_stats", None)
            if stats is not None:  # share of (flipped-space) positives boosted, per label
                fraction = stats["boosted_fraction"].detach()
                boosted = fraction if boosted is None else boosted + fraction
            batches += 1
        criterion.end_epoch()
        stop = time.time() > deadline and epoch < job.epochs
        finite = math.isfinite(total)
        if epoch % job.eval_every and epoch != job.epochs and not stop and finite:
            continue
        val = predict(model, data.x["val"], device)
        with torch.no_grad():
            val_loss = float(criterion(-val.to(device), 1 - data.y["val"].to(device)))
        test = predict(model, data.x["test"], device)
        val_m = task_metrics(val, data.y["val"], data.tasks)
        val_sens = float(np.mean([v for k_, v in val_m.items() if k_.endswith("sens@0.98")]))
        record = {
            "epoch": epoch,
            "train_loss": total,
            "val_loss": val_loss,
            "val_sens": val_sens,
            "ready": criterion.tracker.ready.tolist() if hasattr(criterion, "tracker") else None,
            "seconds": time.perf_counter() - start,
            **{f"val/{k_}": v for k_, v in val_m.items()},
            **{f"test/{k_}": v for k_, v in task_metrics(test, data.y["test"], data.tasks).items()},
        }
        if boosted is not None:
            record["boosted_fraction"] = (boosted / max(batches, 1)).tolist()
        history.append(record)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "seed": seed,
                    **{k_: record[k_] for k_ in ("epoch", "val_loss", "val_sens")},
                }
            ),
            flush=True,
        )
        if epoch in CHECKPOINTS or epoch == job.epochs or stop or not finite:
            saved[f"test_{epoch}"], saved[f"val_{epoch}"] = test, val
        if val_loss < best["val_loss"][0]:
            best["val_loss"] = (val_loss, epoch, test)
        if val_sens > best["val_sens"][0]:
            best["val_sens"] = (val_sens, epoch, test)
        if not finite:
            status = f"failed: non-finite training loss at epoch {epoch}"
            break
        if stop:
            status = f"stopped: deadline after epoch {epoch}"
            break
    return {
        "history": history,
        "best": best,
        "last": test,
        "saved": saved,
        "status": status,
        "seconds": time.perf_counter() - start,
    }


JOBS = {
    "nih": {  # the first chest X-ray study
        "arch": "swav_resnet50",
        "epochs": 200,
        "lr": 1e-4,
        "batch_size": 64,
        "optimizer": "adam",
        "eval_every": 1,
    },
    "chestmnist": {
        "arch": "swav_resnet50",
        "epochs": 15,
        "lr": 1e-4,
        "batch_size": 64,
        "optimizer": "adam",
        "eval_every": 1,
    },
    "pneumoniamnist": {
        "arch": "resnet50",
        "epochs": 100,
        "lr": 1e-4,
        "batch_size": 64,
        "optimizer": "adam",
        "eval_every": 2,
    },
    "creditcard": {
        "arch": "mlp",
        "epochs": 200,
        "lr": 0.01,
        "batch_size": 100,
        "optimizer": "sgd",
        "eval_every": 5,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=list(JOBS))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--arch")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--arms", default="ce,ar_paper")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--data", default="/kaggle/input", help="nih: where the preprocessing output is"
    )
    parser.add_argument("--out", default="benchmarks/results/replication")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    warnings.simplefilter("ignore", UserWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    spec = dict(JOBS[args.job])
    if args.arch:
        spec["arch"] = args.arch
    if args.epochs:
        spec["epochs"] = args.epochs
    if args.job == "creditcard":
        data = load_creditcard()
        pos_weight, window, size = torch.tensor(5.0), 1000, 0
    else:
        if args.job == "nih":
            data = load_nih(Path(args.data), args.size)
        else:
            data = load_medmnist(args.job, args.size)
        y = data.y["train"]
        # Class weights from the class ratio (paper), in the flipped space: normal is the
        # positive class, so pos_weight = #abnormal / #normal per label.
        pos_weight = (y.sum(0) / (len(y) - y.sum(0)).clamp(min=1)).float()
        window, size = None, args.size
    job = Job(args.job, size, pos_weight=pos_weight, window_size=window, **spec)
    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    rows = out / "runs.jsonl"
    done = set()
    if rows.exists():
        done = {json.loads(line)["key"] for line in rows.read_text().splitlines()}
    labels = {f"y_{part}": data.y[part].numpy().astype(np.uint8) for part in ("val", "test")}
    for seed in args.seeds:
        for arm in args.arms.split(","):
            key = f"{args.job}_{size}_{job.arch}_{arm}_{seed}_{job.epochs}"
            if key in done:
                print(f"skip {key}", flush=True)
                continue
            r = train_one(job, data, arm, seed, device)
            selected = {}
            scores = {k: v.numpy().astype(np.float16) for k, v in r["saved"].items()}
            for crit, (score, epoch, test) in r["best"].items():
                scores[f"test_{crit}"] = test.numpy().astype(np.float16)
                selected[crit] = {
                    "epoch": epoch,
                    "score": score,
                    **task_metrics(test, data.y["test"], data.tasks),
                }
            scores["test_last"] = r["last"].numpy().astype(np.float16)
            np.savez(out / "scores" / f"{key}.npz", **scores, **labels)
            row = {
                "key": key,
                "status": r["status"],
                "git_sha": os.environ.get("AUCRESHAPE_GIT_SHA"),
                "job": args.job,
                "size": size,
                "arch": job.arch,
                "arm": arm,
                "config": ARMS[arm],
                "seed": seed,
                "epochs": job.epochs,
                "lr": job.lr,
                "batch_size": job.batch_size,
                "split": {part: len(x) for part, x in data.x.items()},
                "selected": selected,
                "last": task_metrics(r["last"], data.y["test"], data.tasks),
                "history": r["history"],
                "seconds": r["seconds"],
                "package": aucreshape.__version__,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "host": platform.node(),
            }
            with rows.open("a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
