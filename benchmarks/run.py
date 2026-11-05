"""Config-driven benchmark runner.

    uv run python -m benchmarks.run --config benchmarks/configs/tabular.yaml [--dataset NAME]
                                    [--smoke]

For every dataset and seed: pretrain once with BCE, fine-tune every method configuration from
that checkpoint, select the configuration and epoch on validation sensitivity at the target
specificity, and evaluate the selection on the test set. One JSONL row per
(dataset, method, seed), plus the test logits under results/<name>/raw/ for the report.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml

import aucreshape
from benchmarks.evaluation import evaluate
from benchmarks.registry import DATASETS, make_method
from benchmarks.training import TrainConfig, finetune, pretrain


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_dataset(
    name: str, spec: dict[str, Any], config: dict[str, Any], out: Path, smoke: bool
) -> None:
    split = DATASETS[name](**spec.get("params", {}))
    train = TrainConfig(**spec.get("train", {}))
    target = float(spec.get("target", 0.98))
    train = dataclasses.replace(train, select_specificity=target)
    if smoke:
        train = dataclasses.replace(train, pretrain_epochs=1, finetune_epochs=1)
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    methods = {m: (grid[:1] if smoke else grid) for m, grid in config["methods"].items()}
    specs = config.get("specificities", [0.95, 0.98])
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    meta = {
        "package": aucreshape.__version__,
        "git": git_sha(),
        "torch": torch.__version__,
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "device": train.device,
    }
    print(split.describe(), flush=True)
    with (out / f"{name}.jsonl").open("a") as f:
        for seed in seeds:
            state = pretrain(split, seed, train)
            for method_name, grid in methods.items():
                best: dict[str, Any] | None = None
                tried = []
                start = time.perf_counter()
                for params in grid:
                    method = make_method(method_name, dict(params), target)
                    r = finetune(split, seed, state, method, train)
                    tried.append({"params": params, "val": r["val_select"], "epoch": r["epoch"]})
                    if best is None or r["val_select"] > best["val_select"]:
                        best = {**r, "params": params}
                assert best is not None
                metrics = evaluate(
                    best["val_logits"], split.y_val, best["test_logits"], split.y_test, specs
                )
                row = {
                    "dataset": name,
                    "method": method_name,
                    "seed": seed,
                    "selected": best["params"],
                    "epoch": best["epoch"],
                    "val_select": best["val_select"],
                    "grid": tried,
                    "tuning_seconds": time.perf_counter() - start,
                    **metrics,
                    **meta,
                    "target": target,
                    "smoke": smoke,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                torch.save(best["test_logits"], raw / f"{name}_{method_name}_{seed}.pt")
            print(f"[{name}] seed {seed} done", flush=True)
    torch.save(split.y_test, raw / f"{name}_y_test.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", action="append", help="run only these datasets")
    parser.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch, 1 config")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    warnings.simplefilter("ignore", UserWarning)
    config = yaml.safe_load(Path(args.config).read_text())
    out = Path("benchmarks/results") / (config["name"] + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in config["datasets"].items():
        if args.dataset and name not in args.dataset:
            continue
        run_dataset(name, spec or {}, config, out, args.smoke)


if __name__ == "__main__":
    main()
