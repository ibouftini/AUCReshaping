"""The older studies: first chest X-ray study, early benchmark, first credit-card run (exploratory).

Direct paired contrasts, validation-threshold results and the noise floor, from saved scores and
without training::

    python -m benchmarks.review.reanalysis_older --out DIR

Chest X-ray has 2 seeds, which cannot give an interval over seeds, and its saved records hold no
patient identifiers. Its interval therefore covers test sampling only, by image, and carries no
claim. The other two use the two-level interval over one test set each.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.review import analysis, reanalysis, stats

TARGET = 0.98
PAPER, DISEASE = "normal = 1 (paper)", "disease = 1"
E4_TASKS = {"effusion": 2, "pneumothorax": 7, "abnormal": 14}
E4_RULES = {
    "highest validation sensitivity": "test_val_sens",
    "lowest validation loss": "test_val_loss",
    "epoch 200": "test_last",
}
E4_ARMS = ("random_boost", "focal", "ce")
E4_INTERVAL = "test sampling only, by image; 2 seeds"
E4_CLAIM = "none: 2 seeds give no interval over seeds"
S09_MODES = ("prob", "logit", "weight")
S09_ARMS = ("bce", "weighted_bce")
S11_RULES = ("last", "val_loss", "val_sens")


def _fn(specificity: float) -> stats.Metric:
    return lambda scores, y: analysis.sens_at_spec(scores, y, specificity)


def sampling_only_bootstrap(
    groups: Sequence[stats.PairedGroup],
    n_boot: int = 1000,
    seed: int = 0,
    specificity: float = TARGET,
    level: float = 0.95,
) -> dict[str, Any]:
    """Paired difference (arm ``b`` minus arm ``a``) with an interval over test cases only.

    Parameters
    ----------
    groups : sequence of PairedGroup
        Test labels, shape (n,), and scores of both arms, shape (S, n). Every seed is kept in
        every replicate, so the interval says nothing about seed variance.
    n_boot : int, default 1000
        Replicates; test cases are resampled within each class.
    seed : int, default 0
        Seed of the resampling.
    specificity : float, default 0.98
        Where sensitivity is read on the test ROC.
    level : float, default 0.95
        Interval level.

    Returns
    -------
    dict
        ``delta``, ``lo``, ``hi`` and ``per_seed``, in points.
    """
    fn = _fn(specificity)
    per_seed = [100 * (fn(g.b[s], g.y) - fn(g.a[s], g.y)) for g in groups for s in range(len(g.a))]
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot)
    for r in range(n_boot):
        values = []
        for g in groups:
            idx = stats.stratified_indices(g.y, rng)
            y = g.y[idx]
            values += [fn(g.b[s][idx], y) - fn(g.a[s][idx], y) for s in range(len(g.a))]
        replicates[r] = np.mean(values)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(replicates, [alpha, 1 - alpha])
    return {
        "delta": float(np.mean(per_seed)),
        "lo": 100 * float(lo),
        "hi": 100 * float(hi),
        "per_seed": per_seed,
    }


def _e4_runs(root: Path) -> tuple[list[dict[str, Any]], list[int]]:
    rows = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["status"] == "ok"]
    return rows, sorted({row["seed"] for row in rows})


def _e4_load(root: Path, rows: list[dict[str, Any]]) -> dict[tuple[str, int], Any]:
    return {(r["arm"], r["seed"]): np.load(root / "scores" / f"{r['key']}.npz") for r in rows}


def _oriented(
    logits: np.ndarray, disease: np.ndarray, convention: str
) -> tuple[np.ndarray, np.ndarray]:
    scores = logits.astype(np.float32)
    labels = disease.astype(np.float32)
    return (-scores, 1 - labels) if convention == PAPER else (scores, labels)


def e4_contrasts(root: Path, n_boot: int = 1000) -> list[dict[str, Any]]:
    """AUCReshaping (the paper's setting) minus each other arm, per task, convention and rule."""
    rows, seeds = _e4_runs(root)
    files = _e4_load(root, rows)
    table = []
    for task, col in E4_TASKS.items():
        for convention in (PAPER, DISEASE):
            for rule, key in E4_RULES.items():
                for arm in E4_ARMS:
                    y = None
                    ours, theirs = [], []
                    for seed in seeds:
                        disease = files[("ce", seed)]["y_test"][:, col]
                        a, y = _oriented(files[(arm, seed)][key][:, col], disease, convention)
                        b, _ = _oriented(
                            files[("ar_paper", seed)][key][:, col], disease, convention
                        )
                        theirs.append(a)
                        ours.append(b)
                    assert y is not None
                    group = stats.PairedGroup(y, np.stack(theirs), np.stack(ours))
                    fn = _fn(TARGET)
                    table.append(
                        {
                            "study": "E4",
                            "task": task,
                            "convention": convention,
                            "rule": rule,
                            "arm": arm,
                            "seeds": len(seeds),
                            "ar_sens": float(np.mean([100 * fn(s, y) for s in group.b])),
                            "control_sens": float(np.mean([100 * fn(s, y) for s in group.a])),
                            **sampling_only_bootstrap([group], n_boot=n_boot),
                            "interval": E4_INTERVAL,
                            "claim": E4_CLAIM,
                        }
                    )
    return table


def e4_task_selected(root: Path, task: str = "effusion") -> list[dict[str, Any]]:
    """Test sensitivity at the epoch of highest validation sensitivity on one task alone.

    The runs' own rule averages validation sensitivity over the three tasks. Each epoch's metrics
    are in the history, in the paper's convention only.
    """
    rows, _ = _e4_runs(root)
    table = []
    for arm in sorted({row["arm"] for row in rows}):
        epochs, values = [], []
        for row in sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"]):
            best = max(row["history"], key=lambda h: h[f"val/{task}/sens@0.98"])
            epochs.append(best["epoch"])
            values.append(100 * best[f"test/{task}/sens@0.98"])
        table.append(
            {
                "study": "E4",
                "task": task,
                "convention": PAPER,
                "rule": f"highest validation sensitivity on {task}",
                "arm": arm,
                "per_seed_epoch": epochs,
                "per_seed": values,
                "mean": float(np.mean(values)),
            }
        )
    return table


def e4_validation_threshold(
    root: Path, epochs: Sequence[int] = (5, 10, 20, 50, 100, 150, 200)
) -> list[dict[str, Any]]:
    """Results at a threshold fixed on validation, at the epochs with saved validation scores."""
    rows, seeds = _e4_runs(root)
    files = _e4_load(root, rows)
    table = []
    for arm in sorted({row["arm"] for row in rows}):
        for task, col in E4_TASKS.items():
            for convention in (PAPER, DISEASE):
                for epoch in epochs:
                    if f"val_{epoch}" not in files[(arm, seeds[0])].files:
                        continue
                    sens, spec = [], []
                    for seed in seeds:
                        saved = files[(arm, seed)]
                        val, y_val = _oriented(
                            saved[f"val_{epoch}"][:, col], saved["y_val"][:, col], convention
                        )
                        test, y_test = _oriented(
                            saved[f"test_{epoch}"][:, col], saved["y_test"][:, col], convention
                        )
                        at = reanalysis.validation_threshold(val, y_val, test, y_test)
                        sens.append(100 * at[0])
                        spec.append(100 * at[1])
                    table.append(
                        {
                            "study": "E4",
                            "arm": arm,
                            "task": task,
                            "convention": convention,
                            "epoch": epoch,
                            "sens_at_val_threshold": float(np.mean(sens)),
                            "realised_specificity": float(np.mean(spec)),
                            "per_seed_sens": sens,
                            "per_seed_specificity": spec,
                        }
                    )
    return table


def _two_level(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    target: float,
    n_boot: int,
) -> dict[str, Any]:
    """Two-level estimate (arm ``b`` minus arm ``a``) over one test set, in points."""
    group = stats.PairedGroup(y, a, b)
    fn = _fn(target)
    result = stats.two_level_bootstrap([group], fn, n_boot=n_boot)
    ours = [100 * fn(s, y) for s in b]
    theirs = [100 * fn(s, y) for s in a]
    per_seed = [p - q for p, q in zip(ours, theirs, strict=True)]
    lo, hi = 100 * result.lo, 100 * result.hi
    return {
        "seeds": len(per_seed),
        "ar_sens": float(np.mean(ours)),
        "control_sens": float(np.mean(theirs)),
        "delta": 100 * result.point,
        "lo": lo,
        "hi": hi,
        "paired_sd": float(np.std(per_seed, ddof=1)),
        "per_seed": per_seed,
        "interval": "two-level",
        "claim": reanalysis.claim_label(lo, hi),
    }


def _stack(raw: Path, name: str, method: str) -> np.ndarray:
    paths = sorted(
        raw.glob(f"{name}_{method}_[0-9]*.pt"), key=lambda p: int(p.stem.rsplit("_")[-1])
    )
    if not paths:
        raise FileNotFoundError(f"No scores for {name} {method} in {raw}.")
    return np.stack(
        [torch.load(p, map_location="cpu", weights_only=True).numpy().ravel() for p in paths]
    ).astype(np.float32)


def s09_contrasts(raw: Path, name: str, target: float, n_boot: int = 1000) -> list[dict[str, Any]]:
    """Each AUCReshaping mode minus BCE and minus weighted BCE on one early benchmark dataset.

    It had no random-positive control and fine-tuned for a few epochs from a trained model.
    """
    y = torch.load(raw / f"{name}_y_test.pt", map_location="cpu", weights_only=True).numpy()
    y = y.ravel().astype(np.float32)
    scores = {m: _stack(raw, name, m) for m in S09_ARMS}
    return [
        {
            "study": "early benchmark",
            "dataset": name,
            "setting": mode,
            "arm": arm,
            "target": target,
            **_two_level(y, scores[arm], _stack(raw, name, f"aucreshape_{mode}"), target, n_boot),
        }
        for mode in S09_MODES
        for arm in S09_ARMS
    ]


def s11_contrasts(
    raw: Path, fraud: np.ndarray, n_boot: int = 1000, target: float = TARGET
) -> list[dict[str, Any]]:
    """First credit-card run: AUCReshaping (boost 0.2 at five thresholds) minus class-weighted CE.

    The files hold fraud logits and all use one split; sensitivity is on normals, as in the paper.
    """
    y = 1 - fraud.astype(np.float32)
    table = []
    for rule in S11_RULES:
        arrays = []
        for arm in ("ce", "ar_paper_cc"):
            paths = sorted(raw.glob(f"creditcard_0_mlp_{arm}_*_{rule}.pt"))
            arrays.append(
                -np.stack(
                    [
                        torch.load(p, map_location="cpu", weights_only=True).numpy().ravel()
                        for p in paths
                    ]
                ).astype(np.float32)
            )
        table.append(
            {
                "study": "first credit-card run",
                "dataset": "creditcard",
                "convention": PAPER,
                "rule": rule,
                "arm": "ce",
                "target": target,
                **_two_level(y, arrays[0], arrays[1], target, n_boot),
            }
        )
    return table


def floors(values: Mapping[tuple[str, str], Sequence[float]]) -> list[dict[str, Any]]:
    """Differences between the baseline's repeats, per study and dataset, in points."""
    return [
        {"study": study, "dataset": dataset, **reanalysis.noise_floor(seen)}
        for (study, dataset), seen in values.items()
    ]


def _s09_targets(jsonl: Path) -> tuple[float, list[float]]:
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    target = rows[0]["target"]
    baseline = [100 * r[f"sens@{target:g}"] for r in rows if r["method"] == "bce"]
    return target, baseline


def main() -> None:
    """Write every table of the older studies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    e4 = args.results / "s12" / "e4"
    write = reanalysis.write_csv
    write(e4_task_selected(e4), args.out / "e4_task_selected.csv")
    write(e4_validation_threshold(e4), args.out / "e4_validation_threshold.csv")
    baselines: dict[tuple[str, str], list[float]] = {}
    summary = [json.loads(line) for line in (e4 / "runs.jsonl").read_text().splitlines()]
    ce = [r for r in summary if r["arm"] == "ce"]
    baselines[("E4", "effusion, lowest validation loss")] = [
        100 * r["selected"]["val_loss"]["effusion/sens@0.98"] for r in ce
    ]
    baselines[("E4", "effusion, highest validation sensitivity")] = [
        100 * r["selected"]["val_sens"]["effusion/sens@0.98"] for r in ce
    ]
    table = []
    for folder, names in (
        ("tabular", ("creditcard", "mammography", "protein_homo", "synthetic")),
        ("images", ("breastmnist", "pneumoniamnist")),
    ):
        for name in names:
            target, baseline = _s09_targets(args.results / folder / f"{name}.jsonl")
            baselines[("early benchmark", name)] = baseline
            table += s09_contrasts(args.results / folder / "raw", name, target, args.n_boot)
            print("early benchmark", name, flush=True)
    write(table, args.out / "s09_contrasts.csv")
    from benchmarks.data import creditcard

    split = creditcard(seed=0, positive="fraud")
    write(
        s11_contrasts(
            args.results / "replication_local" / "raw",
            split.y_test.numpy().ravel(),
            args.n_boot,
        ),
        args.out / "s11_contrasts.csv",
    )
    s11 = [json.loads(x) for x in (args.results / "replication_local" / "creditcard.jsonl").open()]
    baselines[("first credit-card run", "last epoch")] = [
        100 * r["last"]["fraud/sens@0.98"] for r in s11 if r["arm"] == "ce"
    ]
    write(floors(baselines), args.out / "noise_floor_older.csv")
    print("first credit-card run", flush=True)
    write(e4_contrasts(e4, args.n_boot), args.out / "e4_contrasts.csv")
    print("E4", flush=True)


if __name__ == "__main__":
    main()
