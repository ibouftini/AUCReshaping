"""First experiment: does AUCReshaping reproduce the paper's qualitative effect?

Protocol (fixed before the final runs; pilots used data seed 0 and model seeds 0-3, the final
runs use a fresh data draw and disjoint seeds): pretrain with BCE per seed, then fine-tune
every method from the same checkpoint with the same optimiser, epochs and batch size; select
the epoch on validation sensitivity at the target specificity.

Run: uv run python -m benchmarks.s05_first_experiment   (about 10 minutes on a laptop CPU)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

import aucreshape
from benchmarks import plots
from benchmarks.data import Split, creditcard, synthetic
from benchmarks.evaluation import auc_fn, evaluate, mean_std, paired_bootstrap, sens_fn
from benchmarks.training import Method, TrainConfig, finetune, pretrain

OUT = Path("benchmarks/results/s05")
FIG = Path("benchmarks/results/figures")
SEEDS = [10, 11, 12, 13, 14]


def run(
    split: Split,
    cfg: TrainConfig,
    methods: list[Method],
    specs: list[float],
    tag: str,
    seeds: list[int],
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {m.name: [] for m in methods}
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    with (OUT / f"{tag}.jsonl").open("w") as f:
        for seed in seeds:
            state = pretrain(split, seed, cfg)
            for m in methods:
                r = finetune(split, seed, state, m, cfg)
                ev = evaluate(r["val_logits"], split.y_val, r["test_logits"], split.y_test, specs)
                row = {
                    "dataset": split.name,
                    "tag": tag,
                    "method": m.name,
                    "seed": seed,
                    "epoch": r["epoch"],
                    "train_seconds": r["train_seconds"],
                    **ev,
                    "history": r["history"],
                    "version": aucreshape.__version__,
                    "torch": torch.__version__,
                }
                f.write(json.dumps(row) + "\n")
                row["test_logits"] = r["test_logits"].numpy()
                results[m.name].append(row)
                torch.save(r["test_logits"], OUT / "raw" / f"{tag}_{m.name}_{seed}.pt")
            print(f"[{tag}] seed {seed} done", flush=True)
    return results


def table(
    results: dict[str, list[dict[str, Any]]],
    y: np.ndarray,
    specs: list[float],
    baseline: str = "bce",
) -> str:
    keys = (
        [f"sens@{s:g}" for s in specs] + [f"sens@{s:g}_valthr" for s in specs] + ["auc", "pauc0.05"]
    )
    lines = ["| method | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, rows in results.items():
        lines.append(
            f"| {name} | " + " | ".join(mean_std([r[k] for r in rows]) for k in keys) + " |"
        )
    lines += [
        "",
        f"Paired difference vs `{baseline}` (mean over seeds, 95% stratified paired "
        "bootstrap CI over the test set; * = CI excludes 0; wins = seeds with a gain):",
        "",
        "| method | " + " | ".join(f"delta sens@{s:g}" for s in specs) + " | delta AUC |",
        "|---" * (len(specs) + 2) + "|",
    ]
    base = [r["test_logits"] for r in results[baseline]]
    for name, rows in results.items():
        if name == baseline:
            continue
        cells = []
        for fn, key in [(sens_fn(s), f"sens@{s:g}") for s in specs] + [(auc_fn, "auc")]:
            d, lo, hi = paired_bootstrap(y, base, [r["test_logits"] for r in rows], fn)
            wins = sum(r[key] > b[key] for r, b in zip(rows, results[baseline], strict=True))
            star = "*" if lo > 0 or hi < 0 else ""
            cells.append(f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star} ({wins}/{len(rows)})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def synthetic_main() -> str:
    split = synthetic(seed=1)
    cfg = TrainConfig(
        pretrain_epochs=30, finetune_epochs=10, finetune_lr=3e-4, select_specificity=0.98
    )
    a = {"target_specificity": 0.98}
    methods = [
        Method("bce"),
        Method("prob b=0.05", "aucreshape", {**a, "boost": 0.05}),
        Method("prob b=0.1", "aucreshape", {**a, "boost": 0.1}),
        Method("prob b=0.2", "aucreshape", {**a, "boost": 0.2}),
        Method("prob b=0.4", "aucreshape", {**a, "boost": 0.4}),
        Method(
            "band [0.9,0.95,0.98]",
            "aucreshape",
            {"target_specificity": [0.9, 0.95, 0.98], "boost": [0.4, 0.2, 0.1]},
        ),
    ]
    specs = [0.95, 0.98]
    res = run(split, cfg, methods, specs, "synthetic", SEEDS)
    y = split.y_test.numpy()
    out = f"### Synthetic, converged pretraining\n\n{split.describe()}\n\n{table(res, y, specs)}\n"

    # Boost sweep figure: delta sens@0.98 vs boost, with CIs.
    boosts = [0.05, 0.1, 0.2, 0.4]
    base = [r["test_logits"] for r in res["bce"]]
    deltas, cis = [], []
    for b in boosts:
        d, lo, hi = paired_bootstrap(
            y, base, [r["test_logits"] for r in res[f"prob b={b:g}"]], sens_fn(0.98)
        )
        deltas.append(d * 100)
        cis.append((lo * 100, hi * 100))
    plots.lines(
        boosts,
        {"delta sens@0.98": deltas},
        FIG / "s05_boost_sweep.png",
        "Boost sweep (synthetic, prob mode, target 0.98)",
        "boost b",
        "delta sensitivity vs BCE (points)",
        errors={"delta sens@0.98": cis},
        zero_line=True,
    )
    seed = SEEDS[0]
    plots.roc_zoom(
        y,
        {
            "BCE fine-tuning": res["bce"][0]["test_logits"],
            "AUCReshaping (prob, b=0.2)": res["prob b=0.2"][0]["test_logits"],
        },
        FIG / "s05_roc_synthetic.png",
        f"Synthetic test ROC, FPR <= 0.1 (seed {seed})",
    )
    hist = res["prob b=0.2"][0]["history"]
    plots.lines(
        [h["epoch"] for h in hist],
        {
            "boosted fraction of positives": [h["boosted_fraction"] for h in hist],
            "boosted share of the loss": [h["boosted_loss_share"] for h in hist],
        },
        FIG / "s05_diagnostics.png",
        "Fine-tuning diagnostics (synthetic, b=0.2)",
        "epoch",
        "fraction",
    )
    out += (
        "\nPer-epoch diagnostics (prob b=0.2, first seed): "
        + json.dumps(
            [
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in h.items()
                    if k
                    in (
                        "epoch",
                        "thresholds",
                        "boosted_fraction",
                        "mean_boost",
                        "boosted_loss_share",
                    )
                }
                for h in hist
            ]
        )
        + "\n"
    )
    return out


def synthetic_undertrained() -> str:
    split = synthetic(seed=1)
    cfg = TrainConfig(pretrain_epochs=8, finetune_epochs=10, finetune_lr=3e-4)
    methods = [
        Method("bce"),
        Method("prob b=0.2", "aucreshape", {"target_specificity": 0.98, "boost": 0.2}),
    ]
    res = run(split, cfg, methods, [0.95, 0.98], "synthetic_undertrained", SEEDS)
    return (
        "### Synthetic, under-trained pretraining (8 epochs)\n\n"
        + table(res, split.y_test.numpy(), [0.95, 0.98])
        + "\n"
    )


def credit() -> str:
    split = creditcard(seed=0, positive="normal")
    cfg = TrainConfig(
        hidden=(64, 32),
        dropout=0.5,
        optimizer="sgd",
        pretrain_lr=0.01,
        pretrain_epochs=10,
        finetune_lr=0.003,
        finetune_epochs=5,
        batch_size=100,
        pos_weight=5.0,
        select_specificity=0.98,
    )
    common = {"min_negatives": 100}
    methods = [
        Method("bce"),
        Method(
            "prob b=0.2 @0.98",
            "aucreshape",
            {**common, "target_specificity": 0.98, "boost": 0.2},
            calibrate=True,
        ),
        Method(
            "paper: 0.2 @[0.90..0.99], sum",
            "aucreshape",
            {
                **common,
                "target_specificity": [0.9, 0.92, 0.95, 0.98, 0.99],
                "boost": 0.2,
                "combine": "sum",
            },
            calibrate=True,
        ),
    ]
    specs = [0.9, 0.95, 0.98, 0.99]
    res = run(split, cfg, methods, specs, "creditcard", [0, 1, 2, 3, 4])
    y = split.y_test.numpy()
    frauds = int((y < 0.5).sum())
    plots.roc_zoom(
        y,
        {
            "BCE fine-tuning": res["bce"][0]["test_logits"],
            "AUCReshaping (paper config)": res["paper: 0.2 @[0.90..0.99], sum"][0]["test_logits"],
        },
        FIG / "s05_roc_creditcard.png",
        "Credit card test ROC, FPR <= 0.1 (seed 0)",
        marks=(0.01, 0.02, 0.05),
    )
    note = ", ".join(
        f"spec {s:g} on {frauds} frauds = at most {int(np.floor((1 - s) * frauds + 1e-9))} "
        "frauds passed"
        for s in specs
    )
    return (
        f"### Credit card (paper convention: normal = positive, fraud = reference)\n\n"
        f"{split.describe()}. {note}.\n\n{table(res, y, specs)}\n"
    )


SECTIONS = {
    "synthetic": synthetic_main,
    "synthetic_undertrained": synthetic_undertrained,
    "creditcard": credit,
}

if __name__ == "__main__":
    import sys

    warnings.simplefilter("ignore", UserWarning)
    for name in sys.argv[1:] or list(SECTIONS):
        text = SECTIONS[name]()
        (OUT / f"{name}.md").write_text(text)
        print(text, flush=True)
