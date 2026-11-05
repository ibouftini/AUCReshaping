"""The synthetic laboratory redone with fine-tuning, on every scenario and level (exploratory).

The first laboratory trained every arm from random initialization, while the method is defined for
fine-tuning. Here each seed pretrains with CE to its best validation loss, and every arm fine-tunes
from that same checkpoint, with the same batch order. There are two regimes:

- ``rare``: the rare class is boosted and there are no class weights, as in the first laboratory;
- ``paper``: the majority class is boosted, with class-ratio weights and thresholds from the
  minority class, as in the paper's chest X-ray and credit-card setups.

This is exploratory: it runs before the public protocol, on seeds kept apart from confirmatory ones.
Tuning reads validation data only, per scenario level, and each grid has six settings. Sensitivity
is read at 98% specificity, at the epoch of highest validation sensitivity.

    python -m benchmarks.review.finetune_lab kernels --user U --out DIR --shards 4 --dataset U/CODE

A second run, ``--grid extended``, goes past both edges of each grid. It was added after the first
run, because tuned arms picked the edges often.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from benchmarks.review import analysis, boost_strength, e1, reanalysis, runner, stats, xray

TARGET = 0.98
PAPER_BOOST = {"target_specificity": [0.90, 0.92, 0.95, 0.98], "boost": [0.4, 0.2, 0.1, 0.1]}
# The paper's boosts under each reading of the floor and of how boosts combine, gentle to strong.
AR_GRID: tuple[dict[str, Any], ...] = (
    {"combine": "band", "min_ratio": 0.75},
    {"combine": "band", "min_ratio": 0.5},
    {"combine": "band", "min_ratio": 0.25},
    {"combine": "band", "min_ratio": 0.1},  # the setting the first evaluation ran
    {"combine": "sum", "min_ratio": 0.25},
    {"combine": "sum", "min_ratio": 0.1},
)
WEIGHT_GRID: tuple[dict[str, Any], ...] = tuple(
    {"mode": "weight", "target_specificity": [0.98], "boost": [b]}
    for b in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
)
BCE_GRID: tuple[float, ...] = (1.25, 1.5, 2.0, 4.0, 8.0, 16.0)
# A tuned arm that picks a grid edge may be held back by the grid, so these go past both edges.
EXTENDED_AR_GRID: tuple[dict[str, Any], ...] = (
    {"combine": "band", "min_ratio": 0.9},
    *AR_GRID,
    {"combine": "sum", "min_ratio": 0.05},
    {"combine": "sum", "min_ratio": 0.02},
)
EXTENDED_WEIGHT_GRID: tuple[dict[str, Any], ...] = tuple(
    {"mode": "weight", "target_specificity": [0.98], "boost": [b]}
    for b in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
)
EXTENDED_BCE_GRID: tuple[float, ...] = (1.1, *BCE_GRID, 32.0, 64.0)
ARMS = (
    "ce",
    "aucreshape",
    "ar_weight",
    "weighted_bce",
    "matched_random",
    "same_boost_random",
    "matched_above",
)


@dataclass(frozen=True)
class Settings:
    """The design, fixed before the runs."""

    seeds: tuple[int, ...] = tuple(range(1900, 1910))
    tune_seeds: tuple[int, ...] = (1910, 1911, 1912)
    pretrain_epochs: int | None = None  # the scenario's own length
    finetune_epochs: int = 20
    finetune_lr: float = 3e-4
    n_boot: int = 1000
    scale: float = 1.0
    ar_grid: tuple[dict[str, Any], ...] = AR_GRID
    weight_grid: tuple[dict[str, Any], ...] = WEIGHT_GRID
    bce_grid: tuple[float, ...] = BCE_GRID
    name: str = "standard"


def grid_settings(grid: str) -> Settings:
    """Return the design with the named grids: ``standard``, or ``extended`` past both edges."""
    if grid == "standard":
        return Settings()
    if grid == "extended":
        return Settings(
            ar_grid=EXTENDED_AR_GRID,
            weight_grid=EXTENDED_WEIGHT_GRID,
            bce_grid=EXTENDED_BCE_GRID,
            name=grid,
        )
    raise ValueError(f"grid must be 'standard' or 'extended'; got {grid!r}.")


@dataclass(frozen=True)
class View:
    """One task in one regime; label 1 is the boosted class. ``class_weight`` has shape ()."""

    x_train: Tensor
    y_train: Tensor
    x_val: Tensor
    y_val: Tensor
    x_test: Tensor
    y_test: Tensor
    class_weight: Tensor


def view(task: runner.Task, regime: str) -> View:
    """Express a task in a regime: ``rare`` as it is, ``paper`` with labels flipped and weights."""
    if regime == "rare":
        return View(
            task.x_train, task.y_train, task.x_val, task.y_val, task.x_test, task.y_test,
            torch.tensor(1.0),
        )  # fmt: skip
    if regime != "paper":
        raise ValueError(f"regime must be 'rare' or 'paper'; got {regime!r}.")
    y = 1 - task.y_train
    return View(
        task.x_train, y, task.x_val, 1 - task.y_val, task.x_test, 1 - task.y_test,
        (1 - y).sum() / y.sum().clamp(min=1),
    )  # fmt: skip


def level_key(level: Any) -> str:
    """Return a level's name, as the first laboratory wrote it."""
    return e1.level_tag(level)


def plan() -> list[tuple[str, Any, str]]:
    """Return every scenario level in both regimes."""
    return [
        (name, level, regime)
        for name, experiment in e1.EXPERIMENTS.items()
        for level in experiment.levels
        for regime in ("rare", "paper")
    ]


def finetune_batch(n_train: int) -> int:
    """Return a batch size that gives about 20 steps per epoch, however small the dataset."""
    return max(64, min(1024, n_train // 20))


def base_spec(name: str, level: Any, seed: int) -> runner.RunSpec:
    """Return the scenario's model and pretraining settings, as in the first laboratory."""
    return e1._spec(e1.EXPERIMENTS[name], level, "ce", {}, seed, None)


def _bce(z: Tensor, y: Tensor, weight: Tensor) -> Tensor:
    return functional.binary_cross_entropy_with_logits(z, y, pos_weight=weight)


def pretrain(
    v: View, spec: runner.RunSpec, seed: int, settings: Settings
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Train with CE, keeping the checkpoint with the lowest validation loss.

    Returns
    -------
    state, info
        The checkpoint, and its epoch and validation loss.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = runner.make_model(spec, v.x_train.shape[1])
    optimizer = runner.make_optimizer(spec, model)
    epochs = settings.pretrain_epochs or spec.epochs
    best = (float("inf"), 0, {k: t.clone() for k, t in model.state_dict().items()})
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(v.x_train), generator=generator)
        for i in range(0, len(order), spec.batch_size):
            idx = order[i : i + spec.batch_size]
            loss = _bce(model(v.x_train[idx]).squeeze(-1), v.y_train[idx], v.class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            value = float(_bce(runner.predict(model, v.x_val), v.y_val, v.class_weight))
            if value < best[0]:
                best = (value, epoch, {k: t.clone() for k, t in model.state_dict().items()})
    return best[2], {"best_epoch": best[1], "val_loss": best[0], "epochs": epochs}


def _loss(arm: str, params: dict[str, Any], v: View, seed: int) -> nn.Module:
    """Build an arm, with the tracker's minimum capped at the reference samples that exist."""
    reference = int((v.y_train < 0.5).sum())
    if arm in ("ce", "weighted_bce"):
        return xray.make_loss(arm, params, v.class_weight, 4096, seed)
    kwargs = dict(PAPER_BOOST) | dict(params)
    specs = kwargs["target_specificity"]
    needed = int(np.ceil(10.0 / (1.0 - max(specs)) - 1e-9))
    kwargs["min_negatives"] = min(needed, reference)
    window = max(4096, 2 * kwargs["min_negatives"])
    name = "aucreshape" if arm == "ar_weight" else arm
    return xray.make_loss(name, kwargs, v.class_weight, window, seed)


def finetune(
    v: View,
    spec: runner.RunSpec,
    state: dict[str, Tensor],
    arm: str,
    params: dict[str, Any],
    seed: int,
    settings: Settings,
) -> dict[str, Any]:
    """Fine-tune one arm from a checkpoint and read it at its best validation epoch.

    The tracker starts from the pretrained model's training scores, so thresholds exist from the
    first step, as they would when fine-tuning a trained model.

    Returns
    -------
    dict
        ``epoch``, ``val_sens``, ``test`` (scores, shape (T,)), ``val_at_start``,
        ``boosted_fraction`` and ``recipient_overlap``.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model = runner.make_model(spec, v.x_train.shape[1])
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.finetune_lr, weight_decay=1e-4)
    loss_fn = _loss(arm, params, v, seed)
    tracker = getattr(loss_fn, "tracker", None)
    if tracker is not None:
        tracker.update(runner.predict(model, v.x_train), v.y_train)
    start = runner.predict(model, v.x_val)
    batch = finetune_batch(len(v.x_train))
    best_sens, best_epoch = -1.0, 0
    best_test = runner.predict(model, v.x_test)
    shares, overlaps = [], []
    for epoch in range(1, settings.finetune_epochs + 1):
        model.train()
        loss_fn.train()
        order = torch.randperm(len(v.x_train), generator=generator)
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            loss = loss_fn(model(v.x_train[idx]).squeeze(-1), v.y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            share = xray._boosted_share(loss_fn)
            if share is not None:
                shares.append(share)
            if isinstance(loss_fn, xray.MatchedBoost) and loss_fn.last_boosted.any():
                overlaps.append(loss_fn.last_overlap)
        val = runner.predict(model, v.x_val)
        sens = analysis.sens_at_spec(val.numpy(), v.y_val.numpy(), TARGET)
        if sens > best_sens:
            best_sens, best_epoch = sens, epoch
            best_test = runner.predict(model, v.x_test)
    return {
        "epoch": best_epoch,
        "val_sens": best_sens,
        "test": best_test,
        "val_at_start": start,
        "boosted_fraction": float(np.mean(shares)) if shares else None,
        "recipient_overlap": float(np.mean(overlaps)) if overlaps else None,
    }


def _prepare(name: str, level: Any, regime: str, seed: int, settings: Settings) -> Any:
    task = e1.EXPERIMENTS[name].task(level, seed, settings.scale)
    v = view(task, regime)
    spec = base_spec(name, level, seed)
    state, info = pretrain(v, spec, seed, settings)
    return v, spec, state, info


def tune(name: str, level: Any, regime: str, settings: Settings) -> dict[str, Any]:
    """Choose each tuned arm's setting by mean best validation sensitivity, on the tuning seeds."""
    prepared = [_prepare(name, level, regime, s, settings) for s in settings.tune_seeds]

    def score(arm: str, params: dict[str, Any]) -> float:
        return float(
            np.mean(
                [
                    finetune(v, spec, state, arm, params, seed, settings)["val_sens"]
                    for (v, spec, state, _), seed in zip(prepared, settings.tune_seeds, strict=True)
                ]
            )
        )

    factors = {f: score("weighted_bce", {"factor": f}) for f in settings.bce_grid}
    return {
        "aucreshape": max(settings.ar_grid, key=lambda p: score("aucreshape", p)),
        "ar_weight": max(settings.weight_grid, key=lambda p: score("ar_weight", p)),
        "weighted_bce": {"factor": max(factors, key=lambda f: factors[f])},
    }


def run_level(name: str, level: Any, regime: str, settings: Settings) -> dict[str, Any]:
    """Tune and evaluate one level in one regime, with every contrast.

    Returns
    -------
    dict
        Sensitivity per arm and seed, in percent, the chosen settings, the direct contrasts with
        the permitted wording, and diagnostics of how strong and how targeted the boosts were.
    """
    start = time.perf_counter()
    chosen = tune(name, level, regime, settings)
    settings_for = {
        "ce": {},
        "aucreshape": chosen["aucreshape"],
        "ar_weight": chosen["ar_weight"],
        "weighted_bce": chosen["weighted_bce"],
        "matched_random": chosen["aucreshape"],
        "same_boost_random": chosen["aucreshape"],
        "matched_above": chosen["aucreshape"],
    }
    tests: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    labels, epochs, shares, overlaps, pretrained, baselines = [], [], {}, {}, [], []
    strength: list[dict[str, float]] = []
    for seed in settings.seeds:
        v, spec, state, info = _prepare(name, level, regime, seed, settings)
        y = v.y_test.numpy().astype(np.float32)
        labels.append(y)
        baselines.append(info)
        for arm in ARMS:
            out = finetune(v, spec, state, arm, settings_for[arm], seed, settings)
            tests[arm].append(out["test"].numpy().astype(np.float32))
            if arm == "aucreshape":
                epochs.append(out["epoch"])
                pretrained.append(
                    analysis.sens_at_spec(
                        runner.predict(_loaded(v, spec, state), v.x_test).numpy(), y, TARGET
                    )
                )
                config = dict(PAPER_BOOST) | chosen["aucreshape"]
                strength.append(
                    boost_strength.measure(
                        out["val_at_start"].numpy(), v.y_val.numpy(), config["target_specificity"],
                        config["boost"], combine=config["combine"], min_ratio=config["min_ratio"],
                    )
                )  # fmt: skip
            shares.setdefault(arm, []).append(out["boosted_fraction"])
            overlaps.setdefault(arm, []).append(out["recipient_overlap"])
    arms = {
        arm: [
            100 * analysis.sens_at_spec(s, y, TARGET)
            for s, y in zip(tests[arm], labels, strict=True)
        ]
        for arm in ARMS
    }
    contrasts = []
    for other in ("ce", "weighted_bce", "matched_random", "same_boost_random", "matched_above"):
        groups = [
            stats.PairedGroup(y, a[None], b[None])
            for y, a, b in zip(labels, tests[other], tests["aucreshape"], strict=True)
        ]
        estimate = stats.two_level_bootstrap(
            groups, lambda s, y: analysis.sens_at_spec(s, y, TARGET), n_boot=settings.n_boot
        )
        lo, hi = 100 * estimate.lo, 100 * estimate.hi
        per_seed = [a - b for a, b in zip(arms["aucreshape"], arms[other], strict=True)]
        contrasts.append(
            {
                "minus": other,
                "delta": 100 * estimate.point,
                "lo": lo,
                "hi": hi,
                "gains": estimate.seeds_with_gain,
                "paired_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                "per_seed": per_seed,
                "claim": reanalysis.claim_label(lo, hi),
            }
        )
    return {
        "name": name,
        "tag": level_key(level),
        "regime": regime,
        "status": "ok",
        "grid": settings.name,
        "chosen": chosen,
        "arms": arms,
        "pretrained_sens": 100 * float(np.mean(pretrained)),
        "contrasts": contrasts,
        "diagnostics": {
            "epochs": epochs,
            "strength": {k: float(np.mean([m[k] for m in strength])) for k in strength[0]},
            "boosted_fraction": {
                arm: float(np.mean(v)) for arm, v in shares.items() if v and v[0] is not None
            },
            "recipient_overlap": {
                arm: float(np.mean(v)) for arm, v in overlaps.items() if v and v[0] is not None
            },
            "pretrain_best_epoch": [b["best_epoch"] for b in baselines],
        },
        "seconds": time.perf_counter() - start,
        "seeds": list(settings.seeds),
        "code": os.environ.get("AUCRESHAPE_CODE", "unrecorded"),
    }


def _loaded(v: View, spec: runner.RunSpec, state: dict[str, Tensor]) -> nn.Module:
    model = runner.make_model(spec, v.x_train.shape[1])
    model.load_state_dict(state)
    return model


def safe_level(job: tuple[str, Any, str], settings: Settings) -> dict[str, Any]:
    """Run one level; a failure is recorded as a result, not raised."""
    name, level, regime = job
    try:
        return run_level(name, level, regime, settings)
    except Exception as error:  # every level counts: a failure is a result
        return {
            "name": name, "tag": str(level), "regime": regime, "status": f"failed: {error!r}",
        }  # fmt: skip


def _work(job: tuple[str, Any, str], grid: str = "standard") -> dict[str, Any]:
    torch.set_num_threads(1)
    return safe_level(job, grid_settings(grid))


def run_shard(index: int, shards: int, workers: int, out: Path, grid: str = "standard") -> None:
    """Run one kernel's levels, appending each to ``levels.jsonl``."""
    out.mkdir(parents=True, exist_ok=True)
    jobs = plan()[index::shards]
    context = multiprocessing.get_context("fork")
    with context.Pool(workers) as pool, (out / "levels.jsonl").open("a") as log:
        for row in pool.imap_unordered(functools.partial(_work, grid=grid), jobs):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row["name"], row["tag"], row["regime"], row["status"][:80], flush=True)


KERNEL = """# A private, offline CPU kernel. The code comes from the attached bundle.
import hashlib, os, subprocess, sys
from pathlib import Path

found = sorted(Path("/kaggle/input").rglob("src/aucreshape/__init__.py"))
bundle = found[0].parents[2]
code = hashlib.sha256((bundle / "code_manifest.txt").read_bytes()).hexdigest()
env = dict(os.environ, PYTHONPATH=f"{{bundle / 'src'}}:{{bundle}}", AUCRESHAPE_CODE=code)
subprocess.run(
    [sys.executable, "-m", "benchmarks.review.finetune_lab", "worker",
     "--index", "{index}", "--shards", "{shards}", "--grid", "{grid}"],
    env=env, check=True,
)
"""


def make_kernel(
    root: Path,
    user: str,
    index: int,
    shards: int,
    datasets: Sequence[str],
    grid: str = "standard",
) -> Path:
    """Write one private, offline CPU kernel; return its folder."""
    slug = f"aucreshape-s13c-lab-{index}"
    if grid != "standard":
        slug = f"aucreshape-s13c-lab-{grid}-{index}"
    folder = root / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.py").write_text(KERNEL.format(index=index, shards=shards, grid=grid))
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": list(datasets),
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


def main() -> None:
    """Write the kernels, or run one worker in a kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    kernels = commands.add_parser("kernels")
    kernels.add_argument("--user", default=os.environ.get("KAGGLE_USER", "KAGGLE_USER"))
    kernels.add_argument("--out", type=Path, required=True)
    kernels.add_argument("--shards", type=int, default=4)
    kernels.add_argument("--dataset", action="append", required=True)
    kernels.add_argument("--grid", default="standard")
    worker = commands.add_parser("worker")
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--shards", type=int, required=True)
    worker.add_argument("--workers", type=int, default=4)
    worker.add_argument("--grid", default="standard")
    args = parser.parse_args()
    if args.command == "kernels":
        for index in range(args.shards):
            print(make_kernel(args.out, args.user, index, args.shards, args.dataset, args.grid))
        return
    run_shard(args.index, args.shards, args.workers, Path("/kaggle/working/results"), args.grid)


if __name__ == "__main__":
    main()
