"""Build Markdown tables and figures from a benchmark run's JSONL and raw test logits.

uv run python -m benchmarks.report benchmarks/results/tabular [--figures benchmarks/results/figures]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks import plots
from benchmarks.evaluation import auc_fn, mean_std, paired_bootstrap, sens_fn

LABELS = {
    "bce": "BCE fine-tuning",
    "weighted_bce": "weighted BCE",
    "focal": "focal loss",
    "libauc_pauc": "LibAUC pAUC (DRO)",
    "aucreshape_prob": "AUCReshaping prob",
    "aucreshape_logit": "AUCReshaping logit",
    "aucreshape_weight": "AUCReshaping weight",
}


def load(folder: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[row["dataset"]][row["method"]].append(row)
    return rows


def logits(folder: Path, dataset: str, method: str, seed: int) -> np.ndarray:
    return torch.load(folder / "raw" / f"{dataset}_{method}_{seed}.pt").numpy()


def dataset_section(
    folder: Path, dataset: str, methods: dict[str, list[dict[str, Any]]], n_boot: int
) -> tuple[str, dict[str, tuple[float, float, float]]]:
    y = torch.load(folder / "raw" / f"{dataset}_y_test.pt").numpy()
    target = methods["bce"][0]["target"]
    seeds = [r["seed"] for r in methods["bce"]]
    key = f"sens@{target:g}"
    cols = ["sens@0.95", "sens@0.98", f"{key}_valthr", "auc", "pauc0.05"]
    lines = [
        f"### {dataset}",
        "",
        f"Test: {int((y > 0.5).sum())} positives, {int((y < 0.5).sum())} negatives; "
        f"target specificity {target:g}; {len(seeds)} seeds.",
        "",
        "| method | " + " | ".join(cols) + f" | delta {key} vs BCE [95% CI] | delta AUC [95% CI] |"
        " selected configs |",
        "|---" * (len(cols) + 4) + "|",
    ]
    base = [logits(folder, dataset, "bce", s) for s in seeds]
    deltas: dict[str, tuple[float, float, float]] = {}
    for method, rows in methods.items():
        rows = sorted(rows, key=lambda r: r["seed"])
        cells = [mean_std([r[c] for r in rows]) for c in cols]
        if method == "bce":
            diff = auc_diff = "-"
        else:
            other = [logits(folder, dataset, method, r["seed"]) for r in rows]
            d, lo, hi = paired_bootstrap(y, base, other, sens_fn(target), n_boot=n_boot)
            a, alo, ahi = paired_bootstrap(y, base, other, auc_fn, n_boot=n_boot)
            wins = sum(
                r[key] > b[key]
                for r, b in zip(rows, sorted(methods["bce"], key=lambda r: r["seed"]), strict=True)
            )
            mark = "**" if lo > 0 or hi < 0 else ""
            ci = f"[{lo * 100:+.2f}, {hi * 100:+.2f}]"
            diff = f"{mark}{d * 100:+.2f} {ci}{mark} ({wins}/{len(rows)})"
            amark = "**" if alo > 0 or ahi < 0 else ""
            auc_diff = f"{amark}{a * 100:+.2f} [{alo * 100:+.2f}, {ahi * 100:+.2f}]{amark}"
            deltas[method] = (d * 100, lo * 100, hi * 100)
        configs = ", ".join(
            sorted({",".join(f"{k}={v}" for k, v in r["selected"].items()) or "-" for r in rows})
        )
        lines.append(
            f"| {LABELS.get(method, method)} | "
            + " | ".join(cells)
            + f" | {diff} | {auc_diff} | {configs} |"
        )
    lines += [
        "",
        "delta in percentage points; **bold** = CI excludes 0; (k/n) = seeds with a gain.",
        "",
    ]
    return "\n".join(lines), deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--figures", type=Path, default=Path("benchmarks/results/figures"))
    parser.add_argument("--boot", type=int, default=500)
    args = parser.parse_args()
    data = load(args.folder)
    sections, all_deltas = [], {}
    for dataset, methods in data.items():
        text, deltas = dataset_section(args.folder, dataset, methods, args.boot)
        sections.append(text)
        all_deltas[dataset] = deltas
        seed = methods["bce"][0]["seed"]
        y = torch.load(args.folder / "raw" / f"{dataset}_y_test.pt").numpy()
        best_ar = max((m for m in deltas if m.startswith("aucreshape")), key=lambda m: deltas[m][0])
        best_base = max(
            (m for m in deltas if not m.startswith("aucreshape")), key=lambda m: deltas[m][0]
        )
        plots.roc_zoom(
            y,
            {
                LABELS["bce"]: logits(args.folder, dataset, "bce", seed),
                LABELS[best_base]: logits(args.folder, dataset, best_base, seed),
                LABELS[best_ar]: logits(args.folder, dataset, best_ar, seed),
            },
            args.figures / f"{args.folder.name}_roc_{dataset}.png",
            f"{dataset}: test ROC, FPR <= 0.1 (seed {seed})",
        )
    plots.forest(
        all_deltas,
        LABELS,
        args.figures / f"{args.folder.name}_forest.png",
        "delta sensitivity at the target specificity vs BCE fine-tuning (95% CI)",
    )
    (args.folder / "report.md").write_text("\n".join(sections))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
