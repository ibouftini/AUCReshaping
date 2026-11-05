"""Sanity check: every boost mode on the first experiment's synthetic setup (3 seeds).

Run: uv run python -m benchmarks.s06_modes
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from benchmarks.data import synthetic
from benchmarks.evaluation import evaluate, mean_std
from benchmarks.training import Method, TrainConfig, finetune, pretrain

SEEDS = [20, 21, 22]

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    torch.set_num_threads(4)
    split = synthetic(seed=1)
    cfg = TrainConfig(pretrain_epochs=30, finetune_epochs=10, finetune_lr=3e-4)
    a = {"target_specificity": 0.98}
    methods = [
        Method("bce"),
        Method("prob (b=0.2)", "aucreshape", {**a, "boost": 0.2}),
        Method("logit (b=1.0)", "aucreshape", {**a, "boost": 1.0, "mode": "logit"}),
        Method("weight (b=2.0)", "aucreshape", {**a, "boost": 2.0, "mode": "weight"}),
        Method("prob, linear_ramp", "aucreshape", {**a, "boost": "ramp"}),
    ]
    from aucreshape import linear_ramp

    methods[-1].kwargs["boost"] = linear_ramp(0.2, 0.4)
    rows: dict[str, list[dict[str, float]]] = {m.name: [] for m in methods}
    for seed in SEEDS:
        state = pretrain(split, seed, cfg)
        for m in methods:
            r = finetune(split, seed, state, m, cfg)
            rows[m.name].append(
                evaluate(r["val_logits"], split.y_val, r["test_logits"], split.y_test, [0.95, 0.98])
            )
    keys = ["sens@0.95", "sens@0.98", "auc"]
    print("| mode | " + " | ".join(keys) + " | delta sens@0.98 vs bce (per seed) |")
    print("|---" * (len(keys) + 2) + "|")
    base = np.array([r["sens@0.98"] for r in rows["bce"]])
    for name, rs in rows.items():
        delta = np.array([r["sens@0.98"] for r in rs]) - base
        cells = " | ".join(mean_std([r[k] for r in rs]) for k in keys)
        print(f"| {name} | {cells} | {', '.join(f'{d:+.4f}' for d in delta)} |")
