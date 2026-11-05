"""Summary and figure of the first chest X-ray study: SwAV ResNet-50, NIH ChestX-ray14, 200 epochs.

    uv run --group bench python -m benchmarks.review.report_e4

Reads ``benchmarks/results/s12/e4/runs.jsonl`` and the float16 test logits in ``scores/``
(collected from Kaggle by ``benchmarks.kaggle.collect``). Writes ``summary.csv`` next to them:
test sensitivity at 98% specificity per task, label convention, checkpoint rule, arm and
seed, and the mean difference from CE in points. Draws ``benchmarks/results/figures/s12e/S12e``:
effusion sensitivity over the epochs (the paper's convention) and the share of training
positives that AUCReshaping boosts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from aucreshape import metrics
from benchmarks.review import figures
from benchmarks.review.figures import LABELS, Theme

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "s12" / "e4"
FIGURES = ROOT / "benchmarks" / "results" / "figures" / "s12e"
TASKS = {"effusion": 2, "pneumothorax": 7, "abnormal": 14}
ARMS = ("ce", "ar_paper", "focal", "random_boost")
NAMES = {
    **LABELS,
    "ar_paper": "AUCReshaping (paper's setting)",
    "random_boost": "control: same boosts, random positives",
}
RULES = {
    "highest validation sensitivity": "test_val_sens",  # the pre-registered primary rule
    "lowest validation loss": "test_val_loss",
    "epoch 200": "test_last",
}
CONVENTIONS = ("normal = 1 (paper)", "disease = 1")
SPEC = 0.98
TICKS = (1, 2, 5, 10, 20, 50, 100, 200)


def load(root: Path = RESULTS) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
    return [r for r in rows if r["status"] == "ok"]


def _sens(logits: np.ndarray, disease: np.ndarray, convention: str) -> float:
    """Test sensitivity at ``SPEC`` in percent; logits and labels are abnormality = 1."""
    z = torch.from_numpy(logits.astype(np.float32))
    y = torch.from_numpy(disease.astype(np.float32))
    if convention == CONVENTIONS[0]:
        z, y = -z, 1 - y  # normal is the class whose sensitivity is measured
    return 100 * float(metrics.sensitivity_at_specificity(z, y, SPEC))


def summary(root: Path = RESULTS) -> list[dict[str, Any]]:
    """One entry per task, convention, rule and arm, paired with CE by seed."""
    rows = load(root)
    seeds = sorted({r["seed"] for r in rows})
    files = {(r["arm"], r["seed"]): np.load(root / "scores" / f"{r['key']}.npz") for r in rows}
    table = []
    for task, col in TASKS.items():
        for convention in CONVENTIONS:
            for rule, key in RULES.items():
                base = None
                for arm in [a for a in ARMS if any(r["arm"] == a for r in rows)]:
                    values = [_sens(files[(arm, s)][key][:, col], files[(arm, s)]["y_test"][:, col],
                                    convention) for s in seeds]  # fmt: skip
                    entry: dict[str, Any] = {"task": task, "convention": convention, "rule": rule,
                                             "arm": arm, "per_seed": values, "delta": None,
                                             "per_seed_delta": None}  # fmt: skip
                    if arm == "ce":
                        base = values
                    elif base is not None:
                        diffs = [v - b for v, b in zip(values, base, strict=True)]
                        entry.update(per_seed_delta=diffs, delta=float(np.mean(diffs)))
                    table.append(entry)
    return table


def write(table: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["task", "convention", "rule", "arm", "per_seed", "delta", "per_seed_delta"]
        )
        for r in table:
            delta = "" if r["delta"] is None else f"{r['delta']:.2f}"
            per = " / ".join(f"{v:.1f}" for v in r["per_seed"])
            dper = (
                ""
                if r["per_seed_delta"] is None
                else " / ".join(f"{d:+.1f}" for d in r["per_seed_delta"])
            )
            writer.writerow([r["task"], r["convention"], r["rule"], r["arm"], per, delta, dper])


def curve(
    rows: list[dict[str, Any]], arm: str, field: str, column: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds, in percent, of a per-epoch history field (``column`` for vectors)."""
    runs = [r["history"] for r in rows if r["arm"] == arm]
    epochs = np.array([h["epoch"] for h in runs[0]])
    values = np.array(
        [[h[field] if column is None else h[field][column] for h in run] for run in runs]
    )
    return epochs, 100 * values.mean(0)


def figure(theme: Theme, rows: list[dict[str, Any]]) -> plt.Figure:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    colours = dict.fromkeys(("ar_paper", "random_boost"), theme.roles["aucreshape"])
    colours |= {"ce": theme.roles["ce"], "focal": theme.roles["focal"]}
    for arm in ARMS:
        epochs, mean = curve(rows, arm, "test/effusion/sens@0.98")
        style = {"ls": (0, (4, 2)), "lw": 0.9} if arm == "random_boost" else {"lw": 1.1}
        left.plot(epochs, mean, color=colours[arm], label=NAMES[arm], **style)
    left.set_xlabel("epoch")
    left.set_ylabel(r"test sensitivity at $\mathrm{spec}=0.98$ (\%)")
    left.set_title(
        "Effusion (normal = 1, as in the paper), mean of 2 seeds", fontsize=8, loc="left"
    )
    left.legend(fontsize=7, loc="lower left")
    for arm in ("ar_paper", "random_boost"):
        epochs, boosted = curve(rows, arm, "boosted_fraction", column=TASKS["effusion"])
        style = {"ls": (0, (4, 2)), "lw": 0.9} if arm == "random_boost" else {"lw": 1.1}
        right.plot(epochs, boosted, color=colours[arm], **style)
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xticks(TICKS, [str(t) for t in TICKS])
        ax.minorticks_off()
    right.set_ylim(0, 100)
    right.set_xlabel("epoch")
    right.set_ylabel(r"training positives boosted (\%)")
    right.set_title("Share of positives that get a boost", fontsize=8, loc="left")
    fig.tight_layout()
    return fig


def main() -> None:
    table = summary()
    write(table, RESULTS / "summary.csv")
    for r in table:
        if r["delta"] is not None and r["task"] == "effusion":
            print(
                r["convention"],
                "|",
                r["rule"],
                "|",
                r["arm"],
                f"{r['delta']:+.2f}",
                r["per_seed_delta"],
            )
    rows = load()
    for path in figures.render(lambda theme: figure(theme, rows), "S12e", FIGURES):
        print(path)


if __name__ == "__main__":
    main()
