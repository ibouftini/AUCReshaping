"""Reconcile the credit-card baseline with the paper (exploratory, CE only).

The paper reports a baseline sensitivity of 0.73 +/- 0.007 (Table 7 says at 98% specificity, the
text says 99%) and an AUC of 0.98. Earlier runs here gave 0.41 at 98% on another split. This
module varies one unstated setting at a time around the paper's protocol, and repeats the paper's
configuration over several splits, because at 98% specificity only about two frauds set the
threshold. No AUCReshaping arm runs, so nothing here looks at the method's result.

Paper convention throughout: normal = label 1, specificity is measured on fraud, and the model
outputs a normality logit.

The file is self-contained (numpy, pandas, torch) because it is pushed verbatim as a private,
offline Kaggle script kernel; ``make_kernel`` only sets the ``KERNEL_*`` constants below. Kernels
train and save scores. Metrics are computed afterwards with the library's own metric::

    python -m benchmarks.review.cc_reconcile make --user USER --out DIR --shards 5 [--smoke]
    python -m benchmarks.review.cc_reconcile summary --root DIR
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing
import os
import platform
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

KERNEL_SHARD: int | None = None
KERNEL_SHARDS: int = 1
KERNEL_EPOCHS: int | None = None
KERNEL_MAX_JOBS: int | None = None
KERNEL_EXTENSION: bool = False

# Fresh seeds: earlier runs used 0-9, 100-104 and 1000-1002; confirmatory runs start at 2000.
SEEDS = (1500, 1501, 1502, 1503, 1504)
BASE_SPLIT = 1500
REPEAT_SPLITS = (1501, 1502, 1503, 1504, 1505)
SPECIFICITIES = (0.98, 0.99, 0.9999)


@dataclass(frozen=True)
class Config:
    """One training setup; defaults are ``PAPER``.

    The paper states dropout, batch size, ``positive_weights`` and the epochs. It does not state
    the widths, lr, momentum, standardisation, or which class ``positive_weights`` applies to.
    """

    name: str = "paper"
    widths: tuple[int, ...] = (64, 32)
    dropout: float = 0.5
    lr: float = 0.01
    momentum: float = 0.9
    batch_size: int = 100
    pos_weight: float = 5.0
    weighted_class: str = "normal"
    standardise: bool = True
    epochs: int = 200
    checkpoints: tuple[int, ...] = (10, 50, 100, 200)


PAPER = Config()
CONFIGS = (
    PAPER,
    replace(PAPER, name="pw_fraud", weighted_class="fraud"),
    replace(PAPER, name="momentum_0", momentum=0.0),
    replace(PAPER, name="lr_1e-3", lr=1e-3),
    replace(PAPER, name="lr_1e-1", lr=1e-1),
    replace(PAPER, name="wide", widths=(256, 128)),
    replace(PAPER, name="raw", standardise=False),
)


@dataclass(frozen=True)
class Job:
    """One training run."""

    config: Config
    split_seed: int
    seed: int

    @property
    def key(self) -> str:
        """Return the name of the run and of its score file."""
        return f"{self.config.name}_{self.split_seed}_{self.seed}"


def jobs() -> list[Job]:
    """Return the design: each configuration on one split, the paper's on five more."""
    plan = [Job(config, BASE_SPLIT, seed) for config in CONFIGS for seed in SEEDS]
    plan += [Job(PAPER, split, seed) for split in REPEAT_SPLITS for seed in SEEDS]
    return plan


def extension_jobs() -> list[Job]:
    """Return ``momentum_0`` on the five other splits.

    Added after seeing the first 60 runs, and labelled so: ``momentum_0`` and ``lr_1e-3`` have the
    same effective step and raised sensitivity on the base split, and "a simple SGD optimizer" may
    mean no momentum. The first design cannot say whether this effect holds on other splits.
    """
    momentum_0 = next(config for config in CONFIGS if config.name == "momentum_0")
    return [Job(momentum_0, split, seed) for split in REPEAT_SPLITS for seed in SEEDS]


def shard(plan: list[Job], index: int, shards: int) -> list[Job]:
    """Return the jobs of one kernel."""
    return plan[index::shards]


def paper_split(
    fraud: np.ndarray, split_seed: int, n_test: int = 56_962, n_test_fraud: int = 101
) -> tuple[np.ndarray, np.ndarray]:
    """Random split with the counts of the paper's Table 2 and no validation set.

    Parameters
    ----------
    fraud : ndarray of bool, shape (N,)
        Fraud flags.
    split_seed : int
        Seed of the split.
    n_test, n_test_fraud : int
        Test-set size and its number of frauds.

    Returns
    -------
    train, test : ndarray of int, shapes (N - n_test,) and (n_test,)
        Row indices.
    """
    rng = np.random.default_rng(split_seed)
    frauds = rng.permutation(np.flatnonzero(fraud))
    normals = rng.permutation(np.flatnonzero(~fraud))
    n_normal = n_test - n_test_fraud
    test = np.concatenate([frauds[:n_test_fraud], normals[:n_normal]])
    train = np.concatenate([frauds[n_test_fraud:], normals[n_normal:]])
    return rng.permutation(train), rng.permutation(test)


def standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale both arrays, shape (N, D), with the training mean and standard deviation."""
    mean = train.astype(np.float64).mean(0)
    std = train.astype(np.float64).std(0) + 1e-6
    return (
        ((train - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
    )


def weighted_bce(logits: Tensor, normal: Tensor, weight: float, weighted_class: str) -> Tensor:
    """Binary cross-entropy on normality logits with one class up-weighted.

    Parameters
    ----------
    logits : Tensor, shape (N,)
        Normality logits.
    normal : Tensor, shape (N,)
        1 for normal, 0 for fraud.
    weight : float
        Weight of the chosen class; the other has weight 1.
    weighted_class : {"normal", "fraud"}
        The class that receives ``weight``.

    Returns
    -------
    Tensor, shape ()
        The mean loss.
    """
    if weighted_class == "normal":
        weights = 1.0 + (weight - 1.0) * normal
    elif weighted_class == "fraud":
        weights = 1.0 + (weight - 1.0) * (1.0 - normal)
    else:
        raise ValueError(f"weighted_class must be 'normal' or 'fraud'; got {weighted_class!r}.")
    return functional.binary_cross_entropy_with_logits(logits, normal, weight=weights)


def mlp(num_features: int, widths: tuple[int, ...], dropout: float) -> nn.Module:
    """Linear, ReLU and dropout layers, as the paper describes, ending in one logit."""
    layers: list[nn.Module] = []
    size = num_features
    for width in widths:
        layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
        size = width
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


@dataclass
class Result:
    """The outcome of one run. ``scores`` maps an epoch to test normality logits, shape (M,)."""

    status: str
    scores: dict[int, np.ndarray] = field(default_factory=dict)
    train_loss: list[float] = field(default_factory=list)
    seconds: float = 0.0


def train_run(x: Tensor, normal: Tensor, x_test: Tensor, config: Config, seed: int) -> Result:
    """Train one model with SGD and keep the test scores at the checkpoints.

    Parameters
    ----------
    x : Tensor, shape (N, D)
        Training features.
    normal : Tensor, shape (N,)
        Training labels, 1 for normal.
    x_test : Tensor, shape (M, D)
        Test features. They only pass through the model at the checkpoints; nothing is selected
        on them.
    config : Config
        The setup.
    seed : int
        Seed of the initialisation, the dropout and the batch order.

    Returns
    -------
    Result
        Status ``"ok"``, or ``"nonfinite"`` if the loss or the scores stop being finite. A
        diverged run stops early and keeps the checkpoints it had already passed.
    """
    torch.manual_seed(seed)
    batch_order = torch.Generator().manual_seed(seed)
    model = mlp(x.shape[1], config.widths, config.dropout)
    optimiser = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    result = Result("ok")
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=batch_order)
        total, batches = 0.0, 0
        for i in range(0, len(order), config.batch_size):
            idx = order[i : i + config.batch_size]
            loss = weighted_bce(
                model(x[idx]).squeeze(1), normal[idx], config.pos_weight, config.weighted_class
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item()
            batches += 1
        result.train_loss.append(total / batches)
        if not np.isfinite(total):
            result.status = "nonfinite"
            break
        if epoch in config.checkpoints:
            model.eval()
            with torch.inference_mode():
                scores = model(x_test).squeeze(1).numpy().astype(np.float32)
            if not np.isfinite(scores).all():
                result.status = "nonfinite"
                break
            result.scores[epoch] = scores
    result.seconds = time.perf_counter() - start
    return result


# Shared with the forked workers, so the data is not copied per process.
_FEATURES: np.ndarray | None = None
_FRAUD: np.ndarray | None = None
_OUT: Path | None = None


def _init(features: np.ndarray, fraud: np.ndarray, out: Path) -> None:
    global _FEATURES, _FRAUD, _OUT
    _FEATURES, _FRAUD, _OUT = features, fraud, out
    torch.set_num_threads(1)


def _execute(job: Job) -> dict[str, Any]:
    assert _FEATURES is not None
    assert _FRAUD is not None
    assert _OUT is not None
    config = job.config
    row: dict[str, Any] = {
        "key": job.key,
        "config": asdict(config),
        "split_seed": job.split_seed,
        "seed": job.seed,
        "host": platform.node(),
        "torch": torch.__version__,
        "cpus": os.cpu_count(),
    }
    try:
        train, test = paper_split(_FRAUD, job.split_seed)
        x_train, x_test = _FEATURES[train], _FEATURES[test]
        if config.standardise:
            x_train, x_test = standardise(x_train, x_test)
        result = train_run(
            torch.from_numpy(x_train),
            torch.from_numpy(~_FRAUD[train]).float(),
            torch.from_numpy(x_test),
            config,
            job.seed,
        )
    except Exception as error:  # every run counts: a failure is a result
        row.update(status="error", error=repr(error), seconds=0.0, train_loss=[])
        return row
    row.update(
        status=result.status,
        seconds=result.seconds,
        train_loss=result.train_loss,
        n_train=len(train),
        n_test=len(test),
        n_test_fraud=int(_FRAUD[test].sum()),
    )
    if result.scores:
        arrays: dict[str, np.ndarray] = {f"test_{e}": s for e, s in result.scores.items()}
        np.savez_compressed(_OUT / f"{job.key}.npz", y_normal=~_FRAUD[test], **arrays)
    return row


def run_kernel(out: Path = Path("/kaggle/working/results")) -> None:
    """Run this kernel's jobs, one process per CPU up to 4, appending to ``runs.jsonl``."""
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(sorted(glob.glob("/kaggle/input/**/creditcard.csv", recursive=True))[0])
    if len(frame) != 284_807 or int(frame["Class"].sum()) != 492:
        raise RuntimeError("Unexpected credit-card data: expected 284,807 rows and 492 frauds.")
    features = frame.drop(columns="Class").to_numpy(dtype=np.float32)
    fraud = frame["Class"].to_numpy().astype(bool)
    design = extension_jobs() if KERNEL_EXTENSION else jobs()
    plan = shard(design, KERNEL_SHARD or 0, KERNEL_SHARDS)[: KERNEL_MAX_JOBS or None]
    if KERNEL_EPOCHS is not None:
        short = {"epochs": KERNEL_EPOCHS, "checkpoints": (KERNEL_EPOCHS,)}
        plan = [replace(job, config=replace(job.config, **short)) for job in plan]  # type: ignore[arg-type]
    workers = min(4, os.cpu_count() or 1)
    start = time.perf_counter()
    context = multiprocessing.get_context("fork")
    with (
        context.Pool(workers, _init, (features, fraud, out)) as pool,
        (out / "runs.jsonl").open("a") as log,
    ):
        for row in pool.imap_unordered(_execute, plan):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row["key"], row["status"], round(row["seconds"]), flush=True)
    summary = {"shard": KERNEL_SHARD, "workers": workers, "seconds": time.perf_counter() - start}
    (out / "kernel_meta.json").write_text(json.dumps(summary))


def make_kernel(
    root: Path,
    user: str,
    index: int,
    shards: int,
    *,
    smoke: bool = False,
    smoke_epochs: int = 2,
    extension: bool = False,
) -> Path:
    """Write one private, offline CPU script kernel; return its folder."""
    if smoke:
        slug = "aucreshape-s13a-cc-smoke"
    elif extension:
        slug = f"aucreshape-s13a-cc-ext-{index}"
    else:
        slug = f"aucreshape-s13a-cc-{index}"
    folder = root / slug
    folder.mkdir(parents=True, exist_ok=True)
    values = {
        "KERNEL_SHARD: int | None": index,
        "KERNEL_SHARDS: int": shards,
        "KERNEL_EPOCHS: int | None": smoke_epochs if smoke else None,
        "KERNEL_MAX_JOBS: int | None": 4 if smoke else None,
        "KERNEL_EXTENSION: bool": extension,
    }
    code = Path(__file__).read_text()
    for name, value in values.items():
        code, count = re.subn(rf"^{re.escape(name)} = .*$", f"{name} = {value}", code, flags=re.M)
        if count != 1:
            raise RuntimeError(f"Expected one {name} line; found {count}.")
    (folder / "cc_reconcile.py").write_text(code)
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "cc_reconcile.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["mlg-ulb/creditcardfraud"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


def summarise(root: Path, epoch: int = 200) -> list[dict[str, Any]]:
    """Summarise every run under ``root``, per configuration and split, in the paper's direction.

    Parameters
    ----------
    root : Path
        Folder holding ``runs.jsonl`` files and the score files next to them.
    epoch : int, default 200
        The checkpoint to read; the paper reads the end of training and has no validation set.

    Returns
    -------
    list of dict
        One row per (configuration, split): counts of runs and of failed runs, then mean, standard
        deviation and per-run values of the AUC and of sensitivity at ``SPECIFICITIES``.
    """
    from aucreshape import metrics

    groups: dict[tuple[str, int], list[tuple[dict[str, Any], Path]]] = {}
    for path in sorted(root.rglob("runs.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["config"]["name"], row["split_seed"])
            groups.setdefault(key, []).append((row, path.parent / f"{row['key']}.npz"))
    rows = []
    for (name, split_seed), runs in sorted(groups.items()):
        values: dict[str, list[float]] = {}
        failed = 0
        for row, scores_path in sorted(runs, key=lambda item: item[0]["seed"]):
            if row["status"] != "ok":
                failed += 1
                continue
            with np.load(scores_path) as saved:
                scores = torch.from_numpy(saved[f"test_{epoch}"].astype(np.float64))
                labels = torch.from_numpy(saved["y_normal"]).double()
            sens = metrics.sensitivity_at_specificity(scores, labels, list(SPECIFICITIES))
            for spec, value in zip(SPECIFICITIES, sens.tolist(), strict=True):
                values.setdefault(f"sens@{spec:g}", []).append(value)
            values.setdefault("auc", []).append(float(metrics.partial_auc(scores, labels)))
        summary: dict[str, Any] = {
            "config": name,
            "split_seed": split_seed,
            "runs": len(runs),
            "failed": failed,
        }
        for metric, seen in values.items():
            summary[f"{metric}_mean"] = float(np.mean(seen))
            summary[f"{metric}_sd"] = float(np.std(seen, ddof=1)) if len(seen) > 1 else float("nan")
            summary[f"{metric}_per_run"] = seen
        rows.append(summary)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    names = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run this kernel, or the local ``make`` and ``summary`` commands."""
    if KERNEL_SHARD is not None:
        run_kernel()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make")
    make.add_argument("--user", required=True)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--shards", type=int, default=5)
    make.add_argument("--smoke", action="store_true")
    make.add_argument("--extension", action="store_true")
    report = commands.add_parser("summary")
    report.add_argument("--root", type=Path, required=True)
    report.add_argument("--epoch", type=int, default=200)
    args = parser.parse_args()
    if args.command == "make":
        for index in range(1 if args.smoke else args.shards):
            print(
                make_kernel(
                    args.out,
                    args.user,
                    index,
                    args.shards,
                    smoke=args.smoke,
                    extension=args.extension,
                )
            )
    else:
        rows = summarise(args.root, args.epoch)
        _write_csv(rows, args.root / f"summary_epoch{args.epoch}.csv")
        for row in rows:
            print(row["config"], row["split_seed"], f"runs {row['runs']} failed {row['failed']}")
            for metric in ("auc", *(f"sens@{s:g}" for s in SPECIFICITIES)):
                if f"{metric}_mean" in row:
                    print(f"  {metric}: {row[f'{metric}_mean']:.3f} sd {row[f'{metric}_sd']:.3f}")


if __name__ == "__main__":
    main()
