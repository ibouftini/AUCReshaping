"""Ablations: boost magnitude, threshold strategy, combine rule, far-from-target boosting.

Same protocol as the benchmark, 3 seeds, AUCReshaping prob mode unless stated;
everything is compared with BCE fine-tuning from the same checkpoint at the dataset's target
specificity. Run: uv run python -m benchmarks.ablations [dataset ...]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.evaluation import mean_std, paired_bootstrap, sens_fn
from benchmarks.registry import DATASETS
from benchmarks.training import Method, TrainConfig, finetune, pretrain

OUT = Path("benchmarks/results/ablations")
SEEDS = [200, 201, 202]
SETUPS: dict[str, dict[str, Any]] = {
    "synthetic": {
        "params": {"seed": 1},
        "train": TrainConfig(pretrain_epochs=30, finetune_epochs=10, finetune_lr=3e-4),
    },
    "mammography": {
        "params": {},
        "train": TrainConfig(
            pretrain_epochs=60, finetune_epochs=10, finetune_lr=3e-4, batch_size=128
        ),
    },
}


def variants(target: float) -> dict[str, dict[str, Any]]:
    t = {"target_specificity": target}
    out: dict[str, dict[str, Any]] = {}
    for b in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
        out[f"boost {b}"] = {**t, "boost": b}
    for strategy in ("batch", "ema", "epoch"):
        extra = {"min_negatives": 100} if strategy == "batch" else {}
        out[f"strategy {strategy}"] = {**t, "boost": 0.2, "strategy": strategy, **extra}
    specs = [0.9, 0.95, target]
    for combine in ("band", "sum", "max"):
        out[f"combine {combine}"] = {
            "target_specificity": specs,
            "boost": [0.4, 0.2, 0.1],
            "combine": combine,
        }
    out["far: boost at 0.70 only"] = {"target_specificity": 0.70, "boost": 0.2}
    out["far: 0.70 and target"] = {
        "target_specificity": [0.70, target],
        "boost": [0.2, 0.2],
        "combine": "sum",
    }
    return out


def main(datasets: list[str]) -> None:
    warnings.simplefilter("ignore", UserWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        setup = SETUPS[name]
        split = DATASETS[name](**setup["params"])
        cfg: TrainConfig = setup["train"]
        target = cfg.select_specificity
        y = split.y_test.numpy()
        logits: dict[str, list[np.ndarray]] = {"bce": []}
        for v in variants(target):
            logits[v] = []
        for seed in SEEDS:
            state = pretrain(split, seed, cfg)
            logits["bce"].append(
                finetune(split, seed, state, Method("bce"), cfg)["test_logits"].numpy()
            )
            for v, kwargs in variants(target).items():
                r = finetune(split, seed, state, Method(v, "aucreshape", kwargs), cfg)
                logits[v].append(r["test_logits"].numpy())
            print(f"[{name}] seed {seed} done", flush=True)
        fn = sens_fn(target)
        lines = [
            f"### {name} (target specificity {target:g}, {len(SEEDS)} seeds)",
            "",
            f"| variant | sens@{target:g} | delta vs BCE, points [95% CI] |",
            "|---|---|---|",
            f"| BCE fine-tuning | {mean_std([fn(s, y) for s in logits['bce']])} | - |",
        ]
        rows = []
        for v in variants(target):
            d, lo, hi = paired_bootstrap(y, logits["bce"], logits[v], fn, n_boot=300)
            mark = "**" if lo > 0 or hi < 0 else ""
            lines.append(
                f"| {v} | {mean_std([fn(s, y) for s in logits[v]])} | "
                f"{mark}{d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]{mark} |"
            )
            rows.append({"dataset": name, "variant": v, "delta": d, "lo": lo, "hi": hi})
        text = "\n".join(lines) + "\n"
        (OUT / f"{name}.md").write_text(text)
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(text, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(3)
    main(sys.argv[1:] or list(SETUPS))
