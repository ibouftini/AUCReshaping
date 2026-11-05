"""The new chest X-ray pipeline: a single-output model fine-tuned from pretrained weights.

Effusion on NIH ChestX-ray14, from chest X-ray weights that never saw NIH. Every arm shares the
starting weights, model, optimizer, learning rate, batch size, epochs and augmentation; the seed
fixes the head's initialization, the batch order and the augmentation, so arms with one seed are
paired.

Pilot and tuning runs never open the test files. Confirmatory runs save test scores, which enter no
history that selection reads.

Labels follow the paper unless ``boost_class`` says otherwise: the boosted class is label 1, and
normal is the boosted class. The model outputs a disease logit, and the loss sees ``(-z, 1 - y)``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import math
import os
import random
import secrets
import tarfile
import time
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from aucreshape import AUCReshapingLoss, auc_reshaping_loss, boost_values, metrics
from benchmarks.review import analysis
from benchmarks.review.xray_prepare import backbone_from_state, normalise

TARGET = 0.98
PAPER_THRESHOLDS = [0.90, 0.92, 0.95, 0.98]
PAPER_BOOSTS = [0.4, 0.2, 0.1, 0.1]
SPLIT_SEED = 2000
# Fresh seeds, apart from earlier runs. Confirmatory runs start at 2000.
PILOT_SEEDS = (1700, 1701, 1702)
TUNING_SEEDS = (1800, 1801, 1802)
CONFIRMATORY_START = 2000
CONFIRMATORY_SEEDS = 10
MAX_RUNS_PER_KERNEL = 14  # at about 1.15 h per run and two at a time, about 8 h per kernel
HIDDEN_FIELDS = (
    "train_loss",
    "boosted_fraction",
    "boosted_first_100",
    "recipient_overlap",
    "recipient_mass_overlap",
    "extra_mass",
    "capped_share",
    "seconds",
)
HIDDEN_RUN_FIELDS = ("seconds", "images_per_second")
# Gentle to strong; the strengths were measured on the pilot's models, on validation scores.
AR_TUNING_GRID: tuple[dict[str, Any], ...] = (
    {"combine": "band", "min_ratio": 0.75},  # about 1.3x
    {"combine": "band", "min_ratio": 0.25},  # about 3x
    {"combine": "band", "min_ratio": 0.1},  # about 6x, the setting the first evaluation ran
    {"combine": "sum", "min_ratio": 0.05},  # about 15x
)
BCE_TUNING_GRID: tuple[float, ...] = (1.25, 3.0, 6.0, 15.0)
# Added at both grids' gentle end, because AUCReshaping chose that end of its grid. It is as far as
# this end goes: the extra gradient is at most about 11%, and at a floor of 1 it would be CE.
# The paper's boosts at the floor an earlier study ran, about 6x. The paper states no floor. Here
# AUCReshaping and its control differ enough for a targeting test to mean something.
CONTRAST_SETTING: dict[str, Any] = {"combine": "band", "min_ratio": 0.1}
AR_GENTLE_EXTENSION: dict[str, Any] = {"combine": "band", "min_ratio": 0.9}
BCE_GENTLE_EXTENSION = 1.1
MAX_WEIGHT = 100.0  # the largest per-sample weight a matched control may use


@dataclass(frozen=True)
class Data:
    """Images are uint8 of shape (N, H, W); labels are float32 of shape (N,), with disease as 1."""

    x_train: Tensor
    y_train: Tensor
    x_val: Tensor
    y_val: Tensor
    patients_train: np.ndarray
    x_test: Tensor | None = None
    y_test: Tensor | None = None
    patients_test: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    no_finding_test: Tensor | None = None


def patient_split(
    patients: np.ndarray, fraction: float = 0.1, split_seed: int = SPLIT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Carve validation out by patient.

    Parameters
    ----------
    patients : ndarray, shape (N,)
        Patient id per image.
    fraction : float, default 0.1
        Share of patients in validation.
    split_seed : int
        Seed of the split.

    Returns
    -------
    train, val : ndarray of int
        Image indices. No patient appears in both.
    """
    unique = np.unique(patients)
    chosen = np.random.default_rng(split_seed).permutation(unique)[: round(fraction * len(unique))]
    in_val = np.isin(patients, chosen)
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)


def _column(saved: Any, task: str) -> Tensor:
    index = saved["columns"].tolist().index(task)
    return torch.from_numpy(saved["labels"][:, index].astype(np.float32))


def load_data(root: Path, task: str, include_test: bool, split_seed: int = SPLIT_SEED) -> Data:
    """Load one task. Unless ``include_test`` is set, the test files are never opened."""
    images = np.load(root / "images_trainval.npy", mmap_mode="r")
    with np.load(root / "labels_trainval.npz") as saved:
        y = _column(saved, task)
        patients = saved["patient"]
    train, val = patient_split(patients, split_seed=split_seed)
    x_test = y_test = no_finding_test = None
    patients_test = np.zeros(0, dtype=np.int64)
    if include_test:
        x_test = torch.from_numpy(np.load(root / "images_test.npy"))
        with np.load(root / "labels_test.npz") as saved:
            y_test = _column(saved, task)
            no_finding_test = _column(saved, "No Finding")
            patients_test = saved["patient"]
    return Data(
        x_train=torch.from_numpy(np.asarray(images[train])),
        y_train=y[train],
        x_val=torch.from_numpy(np.asarray(images[val])),
        y_val=y[val],
        patients_train=patients[train],
        x_test=x_test,
        y_test=y_test,
        no_finding_test=no_finding_test,
        patients_test=patients_test,
    )


class MatchedBoost(nn.Module):
    """A control with the same amount of extra gradient as AUCReshaping, on other recipients.

    The same boosts on other samples give a different amount of extra gradient, because Eq. 1
    depends on the recipient's probability. This control measures the extra gradient the targeted
    arm would give in this batch, and gives exactly that much to its own recipients as a constant
    weight, so a comparison with AUCReshaping tests who receives the boost, not how much.

    Parameters
    ----------
    base : AUCReshapingLoss
        The targeted configuration. Its tracker, boosts, mode and class weights are used.
    rule : {"random", "above"}
        ``random``: as many random positives as are flagged. ``above``: only positives the
        thresholds would not flag, the highest-contrast choice.
    seed : int
        Seed of the recipients; it does not touch the global random stream.
    """

    def __init__(self, base: AUCReshapingLoss, rule: str, seed: int) -> None:
        super().__init__()
        if rule not in ("random", "above"):
            raise ValueError(f"rule must be 'random' or 'above'; got {rule!r}.")
        self.base, self.rule = base, rule
        self.tracker = base.tracker
        self.generator = torch.Generator().manual_seed(seed + 7919)
        self.last_boosted = torch.zeros(0, dtype=torch.bool)
        self.last_flagged = torch.zeros(0, dtype=torch.bool)
        self.last_positive = torch.zeros(0, dtype=torch.bool)
        self.last_overlap = 0.0
        self.last_mass_overlap = 0.0
        self.last_capped = False
        self.last_extra = 0.0
        self.last_delivered = 0.0

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the loss for logits and targets of shape (N,)."""
        if not (self.training and torch.is_grad_enabled()):
            return self.base(logits, targets)  # type: ignore[no-any-return]
        base = self.base
        z, y = logits.detach(), targets
        positive = y > 0.5
        thresholds = self.tracker.active_thresholds()
        b = boost_values(z, y, thresholds, base._boost(), combine=base.combine).reshape(z.shape)
        flagged = (b > 0) & positive
        extra = targeted_extra(base, z, y, thresholds)
        boosted = torch.zeros_like(positive)
        if self.rule == "random":
            eligible = torch.nonzero(positive, as_tuple=True)[0]
            order = torch.randperm(len(eligible), generator=self.generator).to(eligible.device)
            boosted[eligible[order[: int(flagged.sum())]]] = True
        else:
            boosted = positive & ~flagged
        own_each = 1.0 - torch.sigmoid(z)
        own = own_each[boosted].sum()
        strength = extra / own if bool(own > 0) else torch.zeros_like(extra)
        self.last_capped = bool(strength + 1.0 > MAX_WEIGHT)
        weights = torch.ones_like(z)
        weights[boosted] = (1.0 + strength).clamp(max=MAX_WEIGHT)
        self.last_extra = float(extra)
        self.last_delivered = float(((weights - 1.0) * own_each)[boosted].sum())
        self.last_mass_overlap = float(own_each[boosted & flagged].sum() / own.clamp(min=1e-12))
        pos_weight = base.pos_weight if base.pos_weight is not None else base._pos_weight_value
        loss = functional.binary_cross_entropy_with_logits(
            logits, y, weight=weights.detach(), pos_weight=_as_tensor(pos_weight, logits)
        )
        self.last_boosted, self.last_flagged, self.last_positive = boosted, flagged, positive
        self.last_overlap = float((boosted & flagged).float().sum() / max(1, int(boosted.sum())))
        self.tracker.update(z, y)
        return loss


def targeted_extra(base: AUCReshapingLoss, z: Tensor, y: Tensor, thresholds: Tensor) -> Tensor:
    """Return the extra gradient mass AUCReshaping gives label 1 in a batch, relative to BCE."""
    grad_z = z.clone().requires_grad_(True)
    with torch.enable_grad():
        loss = auc_reshaping_loss(
            grad_z, y, thresholds, base._boost(), mode=base.mode, combine=base.combine,
            reduction="sum", min_ratio=base.min_ratio,
        )  # fmt: skip
        gradient = torch.autograd.grad(loss, grad_z)[0]
    return (gradient.abs() - (1.0 - torch.sigmoid(z))).clamp(min=0.0)[y > 0.5].sum()


def _as_tensor(value: Any, like: Tensor) -> Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


class PlainBCE(nn.Module):
    """Weighted cross-entropy: CE when ``weight`` is the class ratio, weighted BCE when tuned."""

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the loss for logits and targets of shape (N,)."""
        return functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.weight)


def make_loss(
    arm: str, params: dict[str, Any], class_weight: Tensor, window_size: int, seed: int
) -> nn.Module:
    """Build one arm's loss.

    Parameters
    ----------
    arm : str
        ``ce``, ``weighted_bce``, ``aucreshape``, ``matched_random``, ``matched_above`` or
        ``same_boost_random``.
    params : dict
        ``factor`` for ``weighted_bce``. For the others, the keyword arguments of the loss, with
        ``target_specificity`` and ``boost`` defaulting to the paper's setting.
    class_weight : Tensor, shape ()
        The class-ratio ``pos_weight``, in the space the loss sees.
    window_size : int
        The threshold tracker's window.
    seed : int
        Seed of a control's recipients.

    Returns
    -------
    nn.Module
        The loss.
    """
    if arm == "ce":
        return PlainBCE(class_weight)
    if arm == "weighted_bce":
        return PlainBCE(class_weight * float(params["factor"]))
    if arm not in ("aucreshape", "matched_random", "matched_above", "same_boost_random"):
        raise ValueError(f"Unknown arm {arm!r}.")
    kwargs = dict(params)
    specs = kwargs.pop("target_specificity", PAPER_THRESHOLDS)
    boost = kwargs.pop("boost", PAPER_BOOSTS)
    base = AUCReshapingLoss(
        specs, boost=boost, pos_weight=class_weight, window_size=window_size, **kwargs
    )
    if arm == "aucreshape":
        return base
    if arm == "same_boost_random":
        from benchmarks.replication import RandomBoost

        return RandomBoost(base, seed)
    return MatchedBoost(base, rule=arm.split("_")[1], seed=seed)


ARM_EXAMPLES: dict[str, dict[str, Any]] = {
    "ce": {},
    "weighted_bce": {"factor": 4.0},
    "aucreshape": {"min_negatives": 10},
    "matched_random": {"min_negatives": 10},
    "matched_above": {"min_negatives": 10},
    "same_boost_random": {"min_negatives": 10},
}


@dataclass(frozen=True)
class Run:
    """One training run."""

    arm: str
    seed: int
    lr: float
    weights: str
    params: dict[str, Any] = field(default_factory=dict)
    epochs: int = 10
    batch_size: int = 64
    sampler: bool = False
    boost_class: str = "normal"
    evaluate_test: bool = False
    window_size: int = 4096
    tag: str = ""
    code: str = ""  # a coded run hides its arm until the analysis is frozen

    @property
    def key(self) -> str:
        """Return the name of the run and of its score file."""
        if self.code:
            return f"{self.weights}_{self.code}_{self.seed}"
        settings = json.dumps(self.params, sort_keys=True)
        digest = abs(hash_text(f"{settings}|{self.lr}|{self.sampler}|{self.tag}")) % 10**8
        return f"{self.weights}_{self.arm}_{self.boost_class}_{self.seed}_{digest:08d}"


def hash_text(text: str) -> int:
    """Return a hash that is stable between processes, unlike Python's own."""
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % (2**61 - 1)
    return value


def epoch_order(y: Tensor, sampler: bool, generator: torch.Generator) -> Tensor:
    """Return one epoch of indices, oversampling disease to about half when ``sampler`` is set."""
    if not sampler:
        return torch.randperm(len(y), generator=generator)
    weights = torch.where(y > 0.5, 0.5 / y.sum().clamp(min=1), 0.5 / (1 - y).sum().clamp(min=1))
    return torch.multinomial(weights, len(y), replacement=True, generator=generator)


def _scores(model: nn.Module, x: Tensor, device: str, batch: int = 256) -> Tensor:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            images = normalise(x[i : i + batch].to(device, non_blocking=True))
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                out.append(model(_batch_channels(images)).float().squeeze(1).cpu())
    return torch.cat(out)


def _batch_channels(images: Tensor) -> Tensor:
    return images.unsqueeze(1) if images.dim() == 3 else images


def _boosted_share(loss: nn.Module) -> float | None:
    if isinstance(loss, AUCReshapingLoss):
        return float(loss.last_stats["boosted_fraction"].mean())
    if isinstance(loss, MatchedBoost):
        return float(loss.last_flagged.float().sum() / max(1, int(loss.last_positive.sum())))
    boosted = getattr(loss, "last_stats", None)
    return None if boosted is None else float(boosted["boosted_fraction"].mean())


def _class_weight(y: Tensor) -> Tensor:
    """Return the class-ratio ``pos_weight`` for labels with the boosted class as 1."""
    return (1 - y).sum() / y.sum().clamp(min=1)


def _select(history: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Choose epochs from validation data only."""
    finite = [h for h in history if math.isfinite(h["val_loss"])]
    return {
        "val_sens": int(max(finite, key=lambda h: h["val_sens"])["epoch"]),
        "val_loss": int(min(finite, key=lambda h: h["val_loss"])["epoch"]),
        "last": int(finite[-1]["epoch"]),
    }


def train_run(
    data: Data,
    run: Run,
    build_model: Callable[[int], nn.Module],
    device: str,
    out: Path,
) -> dict[str, Any]:
    """Train one run, saving validation scores per epoch and, if asked, test scores.

    Parameters
    ----------
    data : Data
        The images and labels.
    run : Run
        The settings.
    build_model : callable
        Takes the seed and returns the model, with inputs (N, 1, H, W) and outputs (N, 1).
    device : str
        ``cuda`` or ``cpu``.
    out : Path
        Receives ``{key}.npz``.

    Returns
    -------
    dict
        The run's record, with ``status`` not ``ok`` if training stopped being finite.
    """
    if run.boost_class not in ("normal", "disease"):
        raise ValueError(f"boost_class must be 'normal' or 'disease'; got {run.boost_class!r}.")
    flip = run.boost_class == "normal"
    y_train = (1 - data.y_train) if flip else data.y_train
    y_val = (1 - data.y_val) if flip else data.y_val
    torch.manual_seed(run.seed)
    order_generator = torch.Generator().manual_seed(run.seed)
    augment_generator = torch.Generator().manual_seed(run.seed + 1)
    model = build_model(run.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    criterion = make_loss(
        run.arm, run.params, _class_weight(y_train).to(device), run.window_size, run.seed
    ).to(device)
    history: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {}
    status = "ok"
    start = time.perf_counter()
    trained = 0
    for epoch in range(1, run.epochs + 1):
        model.train()
        criterion.train()
        order = epoch_order(data.y_train, run.sampler, order_generator)
        total, batches, shares, overlaps = 0.0, 0, [], []
        mass_overlaps: list[float] = []
        extra_mass, capped = 0.0, 0
        logs_mass = isinstance(criterion, AUCReshapingLoss | MatchedBoost)
        for i in range(0, len(order), run.batch_size):
            idx = order[i : i + run.batch_size]
            images = data.x_train[idx].to(device, non_blocking=True)
            mirror = (torch.rand(len(idx), generator=augment_generator) < 0.5).to(device)
            images = torch.where(mirror.view(-1, 1, 1), images.flip(-1), images)
            scale = (0.9 + 0.2 * torch.rand(len(idx), 1, 1, generator=augment_generator)).to(device)
            images = (images.float() * scale).clamp(0, 255)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                z = model(_batch_channels(normalise(images))).squeeze(1)
            z = z.float()
            targets = y_train[idx].to(device, non_blocking=True)
            oriented = -z if flip else z
            if isinstance(criterion, AUCReshapingLoss):
                thresholds = criterion.tracker.active_thresholds()
                extra_mass += float(
                    targeted_extra(criterion, oriented.detach(), targets, thresholds)
                )
            loss = criterion(oriented, targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1
            trained += len(idx)
            share = _boosted_share(criterion)
            if share is not None:
                shares.append(share)
            if isinstance(criterion, MatchedBoost):
                extra_mass += criterion.last_delivered
                capped += int(criterion.last_capped)
                if criterion.last_boosted.any():
                    overlaps.append(criterion.last_overlap)
                    mass_overlaps.append(criterion.last_mass_overlap)
        finite = math.isfinite(total)
        val = _scores(model, data.x_val, device) if finite else torch.full_like(y_val, math.nan)
        val_oriented = -val if flip else val
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": total / max(batches, 1),
            "boosted_fraction": float(np.mean(shares)) if shares else None,
            "boosted_first_100": float(np.mean(shares[:100])) if shares else None,
            "recipient_overlap": float(np.mean(overlaps)) if overlaps else None,
            "recipient_mass_overlap": float(np.mean(mass_overlaps)) if mass_overlaps else None,
            "extra_mass": extra_mass if logs_mass else None,
            "capped_share": capped / max(batches, 1) if logs_mass else None,
            "seconds": time.perf_counter() - start,
        }
        if not finite or not torch.isfinite(val).all():
            status = f"failed: not finite at epoch {epoch}"
            record.update(val_loss=math.inf, val_sens=-math.inf, val_auc=math.nan)
            history.append(record)
            break
        criterion.eval()
        with torch.no_grad():
            record["val_loss"] = float(
                functional.binary_cross_entropy_with_logits(
                    val_oriented, y_val, pos_weight=_class_weight(y_train)
                )
            )
        record["val_sens"] = analysis.sens_at_spec(val_oriented.numpy(), y_val.numpy(), TARGET)
        record["val_auc"] = float(metrics.partial_auc(val_oriented, y_val))
        history.append(record)
        saved[f"val_{epoch}"] = val.numpy().astype(np.float32)
        if run.evaluate_test and data.x_test is not None:
            saved[f"test_{epoch}"] = _scores(model, data.x_test, device).numpy().astype(np.float32)
    seconds = time.perf_counter() - start
    row: dict[str, Any] = {
        "key": run.key,
        "status": status,
        "config": asdict(run),
        "history": history,
        "selected": _select(history) if any(math.isfinite(h["val_loss"]) for h in history) else {},
        "seconds": seconds,
        "images_per_second": trained / max(seconds, 1e-9),
        "n_train": len(data.y_train),
        "n_val": len(data.y_val),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "code": os.environ.get("AUCRESHAPE_CODE", "unrecorded"),
    }
    out.mkdir(parents=True, exist_ok=True)
    if run.code:
        row = _seal(row, out)
    arrays = {"y_val": data.y_val.numpy().astype(np.uint8), **saved}
    if run.evaluate_test and data.y_test is not None:
        arrays |= {
            "y_test": data.y_test.numpy().astype(np.uint8),
            "patients_test": data.patients_test,
        }
        if data.no_finding_test is not None:
            arrays["no_finding_test"] = data.no_finding_test.numpy().astype(np.uint8)
    np.savez_compressed(out / f"{run.key}.npz", **arrays)
    return row


def _open_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in ("arm", "params")}


def _write_sealed(key: str, out: Path, content: dict[str, Any]) -> None:
    folder = out / "sealed"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{key}.json").write_text(json.dumps(content))


def _seal(row: dict[str, Any], out: Path) -> dict[str, Any]:
    """Move what would reveal a coded run's arm to a sealed file; return the open record."""
    config = row["config"]
    content = {
        "arm": config["arm"],
        "params": config["params"],
        "history": row["history"],
        **{name: row[name] for name in HIDDEN_RUN_FIELDS},
    }
    _write_sealed(row["key"], out, content)
    history = [{k: v for k, v in h.items() if k not in HIDDEN_FIELDS} for h in row["history"]]
    visible = {k: v for k, v in row.items() if k not in HIDDEN_RUN_FIELDS}
    return {**visible, "config": _open_config(config), "history": history}


def failure_row(job: Run, error: Exception, out: Path) -> dict[str, Any]:
    """Return the record of a run that raised; a coded run's error goes to its sealed file."""
    if not job.code:
        return {"key": job.key, "status": f"failed: {error!r}", "config": asdict(job)}
    content = {"arm": job.arm, "params": job.params, "error": repr(error)}
    _write_sealed(job.key, out, content)
    return {"key": job.key, "status": "failed", "config": _open_config(asdict(job))}


def arm_codes(arms: Iterable[str], salt: str) -> dict[str, str]:
    """Return a code per arm, made of digits, that only the salt can reproduce."""
    return {
        arm: f"arm-{int(hashlib.sha256(f'{salt}|{arm}'.encode()).hexdigest(), 16) % 10**8:08d}"
        for arm in arms
    }


CONFIRMATORY_ARMS = (
    "ce",
    "aucreshape",
    "weighted_bce",
    "aucreshape_6x",
    "matched_random_6x",
    "matched_above_6x",
)


def confirmatory_codes(salt: str) -> dict[str, str]:
    """Return the code of each confirmatory arm. It is read only at unblinding."""
    return arm_codes(CONFIRMATORY_ARMS, salt)


def _check_chosen(chosen: dict[str, dict[str, Any]]) -> None:
    """Refuse a choice that would silently run default settings or a duplicate arm."""
    for arm, needed in (("aucreshape", ("combine", "min_ratio")), ("weighted_bce", ("factor",))):
        if arm not in chosen:
            raise ValueError(f"The chosen settings must include {arm}.")
        for name in needed:
            if name not in chosen[arm]:
                raise ValueError(f"The chosen setting of {arm} must state {name}.")
    if chosen["aucreshape"] == CONTRAST_SETTING:
        raise ValueError("The contrast setting would duplicate the chosen setting.")


def write_chosen(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write the settings that tuning chose, in the form the confirmatory jobs read."""
    table = choose_settings(rows)
    frozen = {arm: table.get(arm, {}).get("chosen") for arm in ("aucreshape", "weighted_bce")}
    for arm, setting in frozen.items():
        if setting is None:
            raise ValueError(f"Tuning chose no setting for {arm}.")
    path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")


def confirmatory_jobs(
    weights: str,
    lr: float,
    chosen: dict[str, dict[str, Any]],
    seeds: Iterable[int],
    salt: str,
) -> list[Run]:
    """Return the six confirmatory arms on every seed, coded, in an order that hides the arms.

    Parameters
    ----------
    weights : str
        The starting weights.
    lr : float
        The learning rate chosen in the pilot.
    chosen : dict
        The tuned settings of ``aucreshape`` and ``weighted_bce``. The three other boosted arms
        use ``CONTRAST_SETTING``.
    seeds : iterable of int
        Confirmatory seeds, from 2000 up.
    salt : str
        Decides the codes and the order. Kept private until the analysis is frozen.

    Returns
    -------
    list of Run
        One run per arm and seed, each reading test data.
    """
    seeds = list(seeds)
    if min(seeds) < CONFIRMATORY_START:
        raise ValueError(f"Confirmatory seeds start at {CONFIRMATORY_START}; got {min(seeds)}.")
    _check_chosen(chosen)
    settings = {
        "ce": {},
        "aucreshape": chosen["aucreshape"],
        "weighted_bce": chosen["weighted_bce"],
        "aucreshape_6x": CONTRAST_SETTING,
        "matched_random_6x": CONTRAST_SETTING,
        "matched_above_6x": CONTRAST_SETTING,
    }
    codes = confirmatory_codes(salt)
    jobs = [
        Run(
            name.removesuffix("_6x"),
            seed,
            lr,
            weights,
            dict(params),
            evaluate_test=True,
            code=codes[name],
        )
        for seed in seeds
        for name, params in settings.items()
    ]
    random.Random(salt).shuffle(jobs)
    return jobs


def confirmatory_kernels(
    root: Path,
    user: str,
    jobs: Sequence[Run],
    datasets: Sequence[str],
    kernels: Sequence[str],
) -> list[Path]:
    """Write the confirmatory runs as enough kernels that none nears the session limit."""
    parts = math.ceil(len(jobs) / MAX_RUNS_PER_KERNEL)
    root.mkdir(parents=True, exist_ok=True)
    (root / "expected_runs.json").write_text(json.dumps([job.key for job in jobs], indent=2) + "\n")
    return [
        make_kernel(root, user, f"confirm-{i}", jobs[i::parts], datasets, kernels)
        for i in range(parts)
    ]


def read_or_create_salt(path: Path) -> str:
    """Return the private salt, creating it once, readable by its owner only."""
    if path.exists():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32)
    path.touch(mode=0o600)
    path.write_text(salt + "\n")
    return salt


def pilot_jobs(weights: str) -> list[Run]:
    """Return the pilot: CE on one set of weights, across lr and with and without the sampler."""
    seed = PILOT_SEEDS[0]
    jobs = [Run("ce", seed, lr, weights) for lr in (1e-3, 1e-4, 1e-5)]
    jobs.append(Run("ce", seed, 1e-4, weights, sampler=True))
    return jobs


def tuning_jobs(weights: str, lr: float) -> list[Run]:
    """Return every setting of both tuned arms on every tuning seed, plus CE, by seed."""
    return [
        job
        for seed in TUNING_SEEDS
        for job in (
            Run("ce", seed, lr, weights),
            *(Run("aucreshape", seed, lr, weights, dict(g)) for g in AR_TUNING_GRID),
            *(Run("weighted_bce", seed, lr, weights, {"factor": f}) for f in BCE_TUNING_GRID),
        )
    ]


def extension_jobs(weights: str, lr: float) -> list[Run]:
    """Return the settings added at the gentle end of both grids, on every tuning seed."""
    return [
        job
        for seed in TUNING_SEEDS
        for job in (
            Run("aucreshape", seed, lr, weights, dict(AR_GENTLE_EXTENSION)),
            Run("weighted_bce", seed, lr, weights, {"factor": BCE_GENTLE_EXTENSION}),
        )
    ]


def tuning_kernels(
    root: Path,
    user: str,
    jobs: Sequence[Run],
    datasets: Sequence[str],
    kernels: Sequence[str],
    parts: int,
) -> list[Path]:
    """Write the tuning runs as several kernels, so that none nears the session limit."""
    return [
        make_kernel(root, user, f"tune-{i}", jobs[i::parts], datasets, kernels)
        for i in range(parts)
    ]


def _gentle_to_strong(arm: str, params: dict[str, Any]) -> tuple[int, float]:
    """Return a sort key: ``sum`` is stronger than ``band``, and a lower floor is stronger."""
    if arm == "weighted_bce":
        return (0, float(params["factor"]))
    return (int(params.get("combine") == "sum"), -float(params.get("min_ratio", 0.1)))


def choose_settings(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Choose each tuned arm's setting from validation data only.

    Parameters
    ----------
    rows : sequence of dict
        Run records. Records that read test data, or from other seeds, are refused.

    Returns
    -------
    dict
        For each tuned arm: ``chosen``, the setting with the highest mean validation sensitivity
        over seeds, at each run's own best epoch; ``settings``, every setting from gentle to strong,
        with its mean and per-seed values, and ``complete``, false if a run failed, which rules it
        out; ``edge``, ``gentle`` or ``strong`` if the choice is at that end of the grid, else None.
    """
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        config = row["config"]
        if config["evaluate_test"]:
            raise ValueError("Tuning must not use runs that read test data.")
        if config["seed"] not in TUNING_SEEDS:
            raise ValueError(f"Tuning uses only tuning seeds; got seed {config['seed']}.")
        settings = grouped.setdefault(config["arm"], {})
        settings.setdefault(json.dumps(config["params"], sort_keys=True), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for arm, settings in grouped.items():
        if arm == "ce":
            continue
        table = []
        ordered = sorted(settings, key=lambda text: _gentle_to_strong(arm, json.loads(text)))
        for text in ordered:
            runs = settings[text]
            values = [
                float(run["history"][run["selected"]["val_sens"] - 1]["val_sens"])
                for run in runs
                if run["status"] == "ok"
            ]
            table.append(
                {
                    "params": json.loads(text),
                    "per_seed": values,
                    "mean_val_sens": float(np.mean(values)) if values else math.nan,
                    "complete": len(values) == len(TUNING_SEEDS),
                }
            )
        eligible = [entry for entry in table if entry["complete"]]
        best = max(eligible, key=lambda entry: entry["mean_val_sens"]) if eligible else None
        edge = None
        if best is not None and len(table) > 1:
            edge = {0: "gentle", len(table) - 1: "strong"}.get(table.index(best))
        out[arm] = {"chosen": best["params"] if best else None, "settings": table, "edge": edge}
    return out


def data_fingerprint(root: Path, include_test: bool) -> dict[str, str]:
    """Return a SHA-256 per prepared data file; test files are read only if ``include_test``."""
    names = ["labels_trainval.npz", "images_trainval.npy"]
    if include_test:
        names += ["labels_test.npz", "images_test.npy"]
    return {name: _file_hash(root / name) for name in names}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def run_jobs(jobs: Sequence[Run], root: Path, weights_dir: Path, out: Path, device: str) -> None:
    """Run jobs in order on one device, appending each record to ``runs.jsonl``."""
    include_test = any(job.evaluate_test for job in jobs)
    data = load_data(root, "Effusion", include_test=include_test)
    fingerprint = data_fingerprint(root, include_test)
    out.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        state = torch.load(weights_dir / f"{job.weights}.pt", map_location="cpu", weights_only=True)

        def build(seed: int, plain: dict[str, Tensor] = state) -> nn.Module:
            return backbone_from_state(plain, seed)

        try:
            row = train_run(data, job, build, device, out)
        except Exception as error:  # every run counts: a failure is a result
            row = failure_row(job, error, out)
        row["data"] = fingerprint
        with (out / "runs.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        print(row["key"], row["status"], round(row.get("images_per_second", 0)), flush=True)


def _embed(features: nn.Module, x: Tensor, device: str, batch: int = 256) -> Tensor:
    parts = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            images = _batch_channels(normalise(x[i : i + batch].to(device)))
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                f = functional.relu(features(images))
                parts.append(functional.adaptive_avg_pool2d(f, 1).flatten(1).float().cpu())
    return torch.cat(parts)


def probe(root: Path, weights_dir: Path, out: Path, device: str, names: Sequence[str]) -> None:
    """Compare candidate weights without fine-tuning, on validation data only.

    For each set, a new linear head is trained on frozen features. Frozen AUC does not predict
    fine-tuned AUC, so this only ranks candidates and catches a bad one. It is not the baseline
    gate.
    """
    data = load_data(root, "Effusion", include_test=False)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        start = time.perf_counter()
        state = torch.load(weights_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        model = backbone_from_state(state, seed=0).to(device).eval()
        train_features = _embed(model.features, data.x_train, device)
        val_features = _embed(model.features, data.x_val, device)
        seconds = time.perf_counter() - start
        mean, std = train_features.mean(0), train_features.std(0).clamp(min=1e-6)
        x_train = ((train_features - mean) / std).to(device)
        x_val = (val_features - mean) / std
        head = nn.Linear(x_train.shape[1], 1).to(device)
        target = data.y_train.to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(500):
            loss = functional.binary_cross_entropy_with_logits(
                head(x_train).squeeze(1), target, pos_weight=_class_weight(target)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            z = head(x_val.to(device)).squeeze(1).cpu()
        row = {
            "weights": name,
            "val_auc": float(metrics.partial_auc(z, data.y_val)),
            "val_sens_normal": analysis.sens_at_spec(-z.numpy(), (1 - data.y_val).numpy(), TARGET),
            "val_sens_disease": analysis.sens_at_spec(z.numpy(), data.y_val.numpy(), TARGET),
            "extract_seconds": seconds,
            "images_per_second": (len(data.y_train) + len(data.y_val)) / max(seconds, 1e-9),
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        }
        with (out / "probe.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


KERNEL = """# A private, offline kernel. The code comes from the attached bundle.
import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path

JOBS = {jobs}
MODE = {mode!r}
found = sorted(Path("/kaggle/input").rglob("src/aucreshape/__init__.py"))
bundle = found[0].parents[2]
code = hashlib.sha256((bundle / "code_manifest.txt").read_bytes()).hexdigest()
env = dict(os.environ, PYTHONPATH=f"{{bundle / 'src'}}:{{bundle}}", AUCRESHAPE_CODE=code)
Path("/tmp/aucreshape_jobs.json").write_text(json.dumps(JOBS))
listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
gpus = max(1, len(listing.splitlines()))
processes = [
    subprocess.Popen(
        [sys.executable, "-m", "benchmarks.review.xray", "worker", "--mode", MODE,
         "--index", str(i), "--workers", str(gpus)],
        env=dict(env, CUDA_VISIBLE_DEVICES=str(i)),
    )
    for i in range(gpus)
]
codes = [p.wait() for p in processes]
print("exit codes", codes, flush=True)
"""


def make_bundle(source: Path, out: Path, user: str, slug: str = "aucreshape-s13c-code") -> Path:
    """Write the code as a private dataset, with a manifest whose hash identifies what ran."""
    files = sorted(
        [
            *(source / "src" / "aucreshape").rglob("*.py"),
            *[
                path
                for path in (source / "benchmarks").rglob("*.py")
                if not {"results", "tests", "__pycache__"} & set(path.relative_to(source).parts)
                and path.name != "publish.py"
            ],
        ]
    )
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(source).as_posix()}"
        for path in files
    ]
    manifest = ("\n".join(lines) + "\n").encode()
    with tarfile.open(out / "code.tar", "w") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(source).as_posix())
        info = tarfile.TarInfo("code_manifest.txt")
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
    meta = {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta) + "\n")
    return out


def _jobs_source(jobs: Sequence[Run]) -> str:
    """Return the job list as source; coded runs are encoded, so opening a kernel hides the arms."""
    payload = [asdict(job) for job in jobs]
    if not any(job.code for job in jobs):
        return repr(payload)
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    return f'json.loads(zlib.decompress(base64.b64decode("{encoded}")))'


def read_kernel_jobs(folder: Path) -> list[dict[str, Any]]:
    """Return the job list a kernel holds. For coded kernels, this unblinds them."""
    line = next(
        text
        for text in (folder / "kernel.py").read_text().splitlines()
        if text.startswith("JOBS = ")
    )
    source = line.removeprefix("JOBS = ")
    if source.startswith("json.loads"):
        encoded = source.split('"')[1]
        loaded: list[dict[str, Any]] = json.loads(zlib.decompress(base64.b64decode(encoded)))
        return loaded
    return ast.literal_eval(source)  # type: ignore[no-any-return]


def make_kernel(
    root: Path,
    user: str,
    name: str,
    jobs: Sequence[Run],
    datasets: Sequence[str],
    kernels: Sequence[str],
    mode: str = "train",
) -> Path:
    """Write one private, offline GPU kernel; return its folder."""
    slug = f"aucreshape-s13c-{name}"
    folder = root / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.py").write_text(KERNEL.format(jobs=_jobs_source(jobs), mode=mode))
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": list(datasets),
        "competition_sources": [],
        "kernel_sources": list(kernels),
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


def _find(pattern: str) -> Path:
    return sorted(Path("/kaggle/input").rglob(pattern))[0]


def main() -> None:
    """Write the bundle or the kernels, or run one worker in a kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("--out", type=Path, required=True)
    bundle.add_argument("--user", default=os.environ.get("KAGGLE_USER", "KAGGLE_USER"))
    probe_kernel = commands.add_parser("probe-kernel")
    probe_kernel.add_argument("--out", type=Path, required=True)
    probe_kernel.add_argument("--user", default=os.environ.get("KAGGLE_USER", "KAGGLE_USER"))
    tune_kernels = commands.add_parser("tune-kernels")
    tune_kernels.add_argument("--out", type=Path, required=True)
    tune_kernels.add_argument("--user", default=os.environ.get("KAGGLE_USER", "KAGGLE_USER"))
    tune_kernels.add_argument("--weights", required=True)
    tune_kernels.add_argument("--lr", type=float, required=True)
    tune_kernels.add_argument("--parts", type=int, default=2)
    tune_kernels.add_argument("--extension", action="store_true")
    choose = commands.add_parser("choose")
    choose.add_argument("--results", type=Path, required=True)
    choose.add_argument("--out", type=Path, required=True)
    confirm = commands.add_parser("confirm-kernels")
    confirm.add_argument("--out", type=Path, required=True)
    confirm.add_argument("--user", default=os.environ.get("KAGGLE_USER", "KAGGLE_USER"))
    confirm.add_argument("--weights", required=True)
    confirm.add_argument("--lr", type=float, required=True)
    confirm.add_argument("--chosen", type=Path, required=True)
    confirm.add_argument("--seeds", type=int, required=True)
    confirm.add_argument("--salt-file", type=Path, required=True)
    unblind = commands.add_parser("write-codes")
    unblind.add_argument("--salt-file", type=Path, required=True)
    unblind.add_argument("--out", type=Path, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--mode", default="train")
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.command == "bundle":
        print(make_bundle(Path(), args.out, args.user))
        return
    if args.command == "probe-kernel":
        sets = ("chex", "mimic_nb", "mimic_ch", "pc")
        jobs = [Run("ce", 0, 0.0, name) for name in sets]
        datasets = (f"{args.user}/aucreshape-s13c-code",)
        kernels = (f"{args.user}/aucreshape-s13c-prepare",)
        print(make_kernel(args.out, args.user, "probe", jobs, datasets, kernels, mode="probe"))
        return
    if args.command == "tune-kernels":
        datasets = (f"{args.user}/aucreshape-s13c-code",)
        kernels = (f"{args.user}/aucreshape-s13c-prepare",)
        if args.extension:
            jobs = extension_jobs(args.weights, args.lr)
            print(make_kernel(args.out, args.user, "tune-ext", jobs, datasets, kernels))
            return
        jobs = tuning_jobs(args.weights, args.lr)
        for folder in tuning_kernels(args.out, args.user, jobs, datasets, kernels, args.parts):
            print(folder)
        return
    if args.command == "choose":
        rows = [
            json.loads(line)
            for path in sorted(args.results.rglob("runs.jsonl"))
            for line in path.read_text().splitlines()
        ]
        write_chosen(rows, args.out)
        return
    if args.command == "confirm-kernels":
        chosen = json.loads(args.chosen.read_text())
        salt = read_or_create_salt(args.salt_file)
        seeds = range(CONFIRMATORY_START, CONFIRMATORY_START + args.seeds)
        jobs = confirmatory_jobs(args.weights, args.lr, chosen, seeds, salt)
        datasets = (f"{args.user}/aucreshape-s13c-code",)
        kernels = (f"{args.user}/aucreshape-s13c-prepare",)
        for folder in confirmatory_kernels(args.out, args.user, jobs, datasets, kernels):
            print(folder)
        return
    if args.command == "write-codes":
        codes = confirmatory_codes(read_or_create_salt(args.salt_file))
        args.out.write_text(json.dumps(codes, indent=2) + "\n")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = _find("images_trainval.npy").parent
    weights_dir = _find("chex.pt").parent
    out = Path("/kaggle/working/results") / f"worker-{args.index}"
    payload = json.loads(Path("/tmp/aucreshape_jobs.json").read_text())
    if args.mode == "probe":
        names = [job["weights"] for job in payload][args.index :: args.workers]
        probe(root, weights_dir, out, device, names)
        return
    jobs = [Run(**job) for job in payload][args.index :: args.workers]
    run_jobs(jobs, root, weights_dir, out, device)


if __name__ == "__main__":
    main()
