"""Runner of the synthetic laboratory: one (task, arm, configuration, seed) run per call.

Every run appends one JSON row to ``<out>/runs.jsonl`` and writes the validation and test
logits of each checkpoint epoch to ``<out>/scores/<key>.npz`` (float16), so every metric can
be recomputed later. Rows carry the spec, its hash, the git SHA, package versions, hardware,
wall time and per-epoch history: training loss, validation loss and sensitivity at the
target, the fraction of positives boosted, and threshold diagnostics at checkpoints.

Pairing: the seed fixes the initialisation, the batch order and dropout, so arms
with the same seed differ only in their loss. Runs are resumable: a finished key is skipped,
and an interrupted run continues from ``<out>/state/<key>.pt`` (saved every ``save_every``
epochs and when ``max_epochs`` is reached), for Kaggle's 12-hour sessions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import aucreshape
from aucreshape import metrics
from benchmarks.review import arms
from benchmarks.training import MLP

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Task:
    """One binary problem, already split. ``oracle`` is a large fresh sample (synthetic)."""

    name: str
    x_train: Tensor
    y_train: Tensor
    x_val: Tensor
    y_val: Tensor
    x_test: Tensor
    y_test: Tensor
    oracle: tuple[Tensor, Tensor] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    task: str
    arm: str
    params: dict[str, Any]
    seed: int
    model: str = "mlp"  # or "linear"
    hidden: tuple[int, ...] = (64, 32)
    dropout: float = 0.0
    optimizer: str = "adam"  # or "sgd" (momentum 0.9)
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 1024
    epochs: int = 50
    checkpoints: tuple[int, ...] = (5, 10, 20, 50)
    target: float = 0.98
    pos_weight: float | None = None
    device: str = "cpu"

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


@cache
def provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    return {
        "git_sha": os.environ.get("AUCRESHAPE_GIT_SHA") or git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "aucreshape": aucreshape.__version__,
        },
        "hardware": {"platform": platform.platform(), "cpus": os.cpu_count(), "gpu": gpu},
    }


def make_model(spec: RunSpec, num_features: int) -> nn.Module:
    if spec.model == "linear":
        return nn.Linear(num_features, 1)
    if spec.model == "mlp":
        return MLP(num_features, spec.hidden, spec.dropout)
    raise ValueError(f"unknown model {spec.model!r}")


def make_optimizer(spec: RunSpec, model: nn.Module) -> torch.optim.Optimizer:
    if spec.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=spec.lr, momentum=0.9, weight_decay=spec.weight_decay
        )
    return torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)


@torch.no_grad()
def predict(model: nn.Module, x: Tensor, batch: int = 8192) -> Tensor:
    model.eval()
    return torch.cat([model(x[i : i + batch]).squeeze(-1) for i in range(0, len(x), batch)])


def _epoch(
    model: nn.Module,
    task: Task,
    loss_fn: arms.Loss,
    opt: torch.optim.Optimizer,
    spec: RunSpec,
    gen: torch.Generator,
) -> tuple[float, float | None]:
    model.train()
    if isinstance(loss_fn, nn.Module):
        loss_fn.train()
    order = torch.randperm(len(task.x_train), generator=gen).to(task.x_train.device)
    losses, boosted = [], []
    for i in range(0, len(order), spec.batch_size):
        idx = order[i : i + spec.batch_size]
        z = model(task.x_train[idx]).squeeze(-1)
        y = task.y_train[idx]
        loss = loss_fn(z, y, idx) if getattr(loss_fn, "needs_index", False) else loss_fn(z, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        frac = arms.boosted_fraction(loss_fn)
        if frac is not None:
            boosted.append(frac)
    return float(np.mean(losses)), (float(np.mean(boosted)) if boosted else None)


def _threshold(z: Tensor, y: Tensor, s: float) -> float:
    return float(metrics.threshold_at_specificity(z, y, s))


def _diagnostics(
    model: nn.Module, task: Task, spec: RunSpec, val: Tensor, test: Tensor, loss_fn: arms.Loss
) -> dict[str, Any]:
    """Where the training, validation and test thresholds at the target sit (H2, H6)."""
    train = predict(model, task.x_train)
    s = spec.target
    thr = _threshold(train, task.y_train, s)
    positive = task.y_train > 0.5
    below = positive & (train < thr)
    out: dict[str, Any] = {
        "train_threshold": thr,
        "val_threshold": _threshold(val, task.y_val, s),
        "test_threshold": _threshold(test, task.y_test, s),
        "below_train_threshold": float(below.sum() / positive.sum().clamp(min=1)),
    }
    if "flipped" in task.meta:  # H3: share of label noise among the positives below
        flipped = torch.as_tensor(task.meta["flipped"], device=below.device)
        out["flipped_share_below"] = float((below & flipped).sum() / below.sum().clamp(min=1))
    tracker = getattr(loss_fn, "tracker", None)
    if tracker is not None:
        out["tracker_thresholds"] = tracker.thresholds.flatten().tolist()
    return out


def _find(path: Path, key: str) -> dict[str, Any] | None:
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["key"] == key:
                return row  # type: ignore[no-any-return]
    return None


def _rng_state(gen: torch.Generator) -> dict[str, Any]:
    state = {"gen": gen.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(gen: torch.Generator, state: dict[str, Any]) -> None:
    gen.set_state(state["gen"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def run(
    task: Task,
    spec: RunSpec,
    out: Path,
    *,
    max_epochs: int | None = None,
    save_every: int = 10,
    deadline: float | None = None,
) -> dict[str, Any] | None:
    """Train one run, or skip it if finished; return its row, or None if stopped early.

    ``deadline`` is a Unix time (default: the ``AUCRESHAPE_DEADLINE`` environment variable,
    set by the Kaggle kernels): after the first epoch that ends past it, the state is saved
    and the run stops, so a follow-up kernel can resume it.
    """
    if deadline is None:
        deadline = float(os.environ.get("AUCRESHAPE_DEADLINE", "inf"))
    out.mkdir(parents=True, exist_ok=True)
    key = spec.key()
    done = _find(out / "runs.jsonl", key)
    if done is not None:
        return done

    dev = torch.device(spec.device)
    task = Task(
        task.name,
        *(t.to(dev) for t in (task.x_train, task.y_train, task.x_val, task.y_val)),
        *(t.to(dev) for t in (task.x_test, task.y_test)),
        oracle=None if task.oracle is None else (task.oracle[0].to(dev), task.oracle[1].to(dev)),
        meta=task.meta,
    )
    torch.manual_seed(spec.seed)
    model = make_model(spec, task.x_train.shape[1]).to(dev)
    opt = make_optimizer(spec, model)
    loss_fn = arms.make_arm(
        spec.arm, spec.params, spec.target, spec.pos_weight, spec.seed, len(task.x_train), str(dev)
    )
    gen = torch.Generator().manual_seed(spec.seed)
    source = getattr(loss_fn, "threshold_source", None)

    def recalibrate() -> None:
        if source == "val":
            arms.calibrate(loss_fn, predict(model, task.x_val), task.y_val)
        elif source == "oracle":
            assert task.oracle is not None, "the oracle arm needs task.oracle"
            arms.calibrate(loss_fn, predict(model, task.oracle[0]), task.oracle[1])

    state_path = out / "state" / f"{key}.pt"
    history: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {}
    start, seconds = 0, 0.0
    if state_path.exists():
        state = torch.load(state_path, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if isinstance(loss_fn, nn.Module):
            loss_fn.load_state_dict(state["loss"])
        _set_rng_state(gen, state["rng"])
        history, saved, start, seconds = (
            state["history"],
            state["scores"],
            state["epoch"],
            state["seconds"],
        )
    else:
        recalibrate()

    def save_state(epoch: int, elapsed: float) -> None:
        state_path.parent.mkdir(exist_ok=True)
        loss_state = loss_fn.state_dict() if isinstance(loss_fn, nn.Module) else {}
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "loss": loss_state,
                "rng": _rng_state(gen),
                "history": history,
                "scores": saved,
                "epoch": epoch,
                "seconds": elapsed,
            },
            state_path,
        )

    status = "ok"
    t0 = time.perf_counter()
    for epoch in range(start + 1, spec.epochs + 1):
        train_loss, boosted = _epoch(model, task, loss_fn, opt, spec, gen)
        if not math.isfinite(train_loss):
            status = f"failed: non-finite training loss at epoch {epoch}"
            break
        recalibrate()
        val = predict(model, task.x_val)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(F.binary_cross_entropy_with_logits(val, task.y_val)),
            "val_sens": float(metrics.sensitivity_at_specificity(val, task.y_val, spec.target)),
            "boosted_fraction": boosted,
        }
        if epoch in spec.checkpoints:
            test = predict(model, task.x_test)
            saved[f"val_{epoch}"] = val.cpu().numpy().astype(np.float16)
            saved[f"test_{epoch}"] = test.cpu().numpy().astype(np.float16)
            record["diagnostics"] = _diagnostics(model, task, spec, val, test, loss_fn)
        history.append(record)
        elapsed = seconds + time.perf_counter() - t0
        out_of_time = max_epochs is not None and epoch - start >= max_epochs
        if (out_of_time or time.time() >= deadline) and epoch < spec.epochs:
            save_state(epoch, elapsed)
            return None
        if save_every and epoch % save_every == 0 and epoch < spec.epochs:
            save_state(epoch, elapsed)

    (out / "scores").mkdir(exist_ok=True)
    np.savez_compressed(out / "scores" / f"{key}.npz", **saved)
    y_train = task.y_train
    row = {
        "key": key,
        "spec": asdict(spec),
        "task": {
            "name": task.name,
            "n_train": len(y_train),
            "pos_train": int((y_train > 0.5).sum()),
            "n_val": len(task.y_val),
            "n_test": len(task.y_test),
            "pos_test": int((task.y_test > 0.5).sum()),
        },
        "status": status,
        "history": history,
        "seconds": seconds + time.perf_counter() - t0,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **provenance(),
    }
    line = json.dumps(row)
    with (out / "runs.jsonl").open("a") as f:
        f.write(line + "\n")
    state_path.unlink(missing_ok=True)
    return json.loads(line)  # the same types as a row read back from disk
