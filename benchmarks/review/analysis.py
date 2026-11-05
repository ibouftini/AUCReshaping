"""Analysis helpers: fast metrics, checkpoint selection, paired effects.

Effects are fractions of sensitivity (0.01 = 1 point); reports convert them to points.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.review import stats


def sens_at_spec(scores: np.ndarray, y: np.ndarray, specificity: float) -> float:
    """``aucreshape.metrics.sensitivity_at_specificity`` (interpolated ROC) in linear time.

    The ROC is read at FPR = x = 1 - specificity. With c the (k+1)-th largest negative score,
    k = floor(x n_neg), the ROC crosses x on the segment through the scores tied at c: it runs
    from (N_gt, P_gt) to (N_gt + N_eq, P_gt + P_eq) (counts above and at c), so the TPR is
    (P_gt + P_eq (x n_neg - N_gt) / N_eq) / n_pos. Without ties this is the count of positives
    above c; float16 scores make ties common, hence the segment.
    """
    negative = scores[y < 0.5]
    positive = scores[y > 0.5]
    n_neg = len(negative)
    x = (1 - specificity) * n_neg
    k = math.floor(x + 1e-9 * n_neg)
    if k >= n_neg:
        return 1.0
    cut = np.partition(negative, n_neg - 1 - k)[n_neg - 1 - k]
    n_gt, n_eq = int((negative > cut).sum()), int((negative == cut).sum())
    p_gt, p_eq = int((positive > cut).sum()), int((positive == cut).sum())
    fraction = min(max((x - n_gt) / n_eq, 0.0), 1.0)
    return (p_gt + p_eq * fraction) / len(positive)


def load_rows(folder: Path) -> list[dict[str, Any]]:
    """Every row under ``folder``, each tagged with the folder that holds its scores."""
    rows = []
    for path in sorted(folder.rglob("runs.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["_dir"] = str(path.parent)
            rows.append(row)
    return rows


def selected_epoch(row: dict[str, Any], rule: str = "val") -> int:
    """Checkpoint chosen by ``rule``: "val" (highest validation sensitivity),
    "loss" (lowest validation loss) or "last"."""
    checkpoints = set(row["spec"]["checkpoints"])
    history = [h for h in row["history"] if h["epoch"] in checkpoints]
    if rule == "last":
        return int(history[-1]["epoch"])
    if rule == "loss":
        return int(min(history, key=lambda h: h["val_loss"])["epoch"])
    return int(max(history, key=lambda h: h["val_sens"])["epoch"])


def scores_at(row: dict[str, Any], epoch: int, folder: Path | None = None) -> np.ndarray:
    path = Path(row.get("_dir", folder or ".")) / "scores" / f"{row['key']}.npz"
    with np.load(path) as f:
        return f[f"test_{epoch}"].astype(np.float32)


def paired_effect(
    rows: list[dict[str, Any]],
    folder: Path | None,
    labels: Callable[[int], np.ndarray],
    arm_a: str,
    arm_b: str,
    specificity: float,
    rule: str = "val",
    epoch: int | None = None,
    n_boot: int = 1000,
) -> stats.Estimate:
    """delta = sens(b) - sens(a) at ``specificity``, paired by seed; each seed has its own test set.

    ``labels(seed)`` returns that seed's test labels. Each arm's checkpoint follows ``rule``
    unless ``epoch`` is given.
    """
    by = {(r["spec"]["arm"], r["spec"]["seed"]): r for r in rows if r["status"] == "ok"}
    groups = []
    for seed in sorted({s for arm, s in by if arm == arm_a} & {s for arm, s in by if arm == arm_b}):
        a, b = by[(arm_a, seed)], by[(arm_b, seed)]
        sa = scores_at(a, epoch or selected_epoch(a, rule), folder)
        sb = scores_at(b, epoch or selected_epoch(b, rule), folder)
        groups.append(stats.PairedGroup(labels(seed), sa[None], sb[None]))
    return stats.two_level_bootstrap(
        groups, lambda s, y: sens_at_spec(s, y, specificity), n_boot=n_boot
    )
