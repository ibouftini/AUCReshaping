"""The synthetic laboratory: eight scenarios, one real situation and one effect each.

    uv run --group bench python -m benchmarks.review.e1 S-1 [S-2 ...] [--workers 10]

Per scenario: the tuned arms are tuned at the centre level on seeds 1000-1002 (the
configuration with the highest mean validation sensitivity at its best checkpoint), then every
arm runs on seeds 0-9 at every level. The data sample depends on the seed, so arms are paired
on data, initialisation and batch order. Results: ``<out>/<scenario>/tune/...`` and
``<out>/<scenario>/<level>/seed<k>/runs.jsonl`` (one folder per job, so parallel workers never
write the same file), plus ``<out>/<scenario>/chosen.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.review import manipulate, runner, synthetic
from benchmarks.review.runner import RunSpec, Task

OUT = Path(__file__).resolve().parents[1] / "results" / "s12" / "e1"
TARGET = 0.98
D = 20
PREVALENCE = 0.05
N_TRAIN, N_TEST = 20_000, 100_000
CHECKPOINTS = (10, 20, 50, 100, 150, 200)
LONG = (*CHECKPOINTS, 300, 400, 500)
TUNED = {
    "aucreshape": [{"boost": b} for b in (0.05, 0.1, 0.2, 0.4)],
    "weighted_bce": [{"pos_weight": w} for w in (2.0, 4.0, 8.0, 16.0)],
    "focal": [{"gamma": g} for g in (0.5, 1.0, 2.0, 3.0)],
}
CONTROLS = ("random_boost", "boost_all", "placebo", "heldout", "oracle")
EVAL_ARMS = ("ce", *TUNED, *CONTROLS)
# Used when a tuned arm picks the largest value of its grid.
EXTRA = {
    "aucreshape": [
        {"boost": 0.6},
        {"boost": 0.8},
        *({"mode": "weight", "boost": b} for b in (2.0, 4.0, 8.0, 16.0)),
    ],
    "weighted_bce": [{"pos_weight": w} for w in (32.0, 64.0, 128.0)],
    "focal": [{"gamma": g} for g in (4.0, 5.0)],
}
TUNE_SEEDS = (1000, 1001, 1002)
SEEDS = tuple(range(10))

LOOK = synthetic.LookAlikes(d=D, share=0.02)  # the centre of S-3 and of S-5 to S-8


@dataclass(frozen=True)
class Experiment:
    name: str
    situation: str
    levels: tuple[Any, ...]
    centre: Any
    build: Callable[[Any, int, float], Task]  # level, seed, scale
    spec: Callable[[Any], dict[str, Any]] = lambda level: {}  # level -> RunSpec fields

    def task(self, level: Any, seed: int, scale: float = 1.0) -> Task:
        return self.build(level, seed, scale)


def _n(n: int, scale: float) -> int:
    return max(200, round(n * scale))


def _plain(scenario_of: Callable[[Any], synthetic.Scenario]) -> Callable[[Any, int, float], Task]:
    def build(level: Any, seed: int, scale: float) -> Task:
        return synthetic.make_task(
            "e1", scenario_of(level), PREVALENCE, _n(N_TRAIN, scale), _n(N_TEST, scale), seed
        )

    return build


def _noisy(level: float, seed: int, scale: float) -> Task:
    task = synthetic.make_task("e1", LOOK, PREVALENCE, _n(N_TRAIN, scale), _n(N_TEST, scale), seed)
    rng = np.random.default_rng(seed + 1)
    y_train, flipped = manipulate.noisy_labels(task.y_train.numpy(), rng, pos_contamination=level)
    y_val, _ = manipulate.noisy_labels(task.y_val.numpy(), rng, pos_contamination=level)
    task.y_train, task.y_val = torch.from_numpy(y_train), torch.from_numpy(y_val)
    task.meta["flipped"] = flipped
    return task


def _rare(level: tuple[int, float], seed: int, scale: float) -> Task:
    n_pos, prevalence = level
    n_train = round(n_pos / prevalence)
    return synthetic.make_task("e1", LOOK, prevalence, n_train, _n(N_TEST, scale), seed)


def _drift(level: float, seed: int, scale: float) -> Task:
    deployed = replace(LOOK, share=level)
    return synthetic.make_task(
        "e1", LOOK, PREVALENCE, _n(N_TRAIN, scale), _n(N_TEST, scale), seed, test_scenario=deployed
    )


def _capacity(level: tuple[str, tuple[int, ...], int], seed: int, scale: float) -> Task:
    return synthetic.make_task("e1", LOOK, PREVALENCE, _n(level[2], scale), _n(N_TEST, scale), seed)


EXPERIMENTS = {
    e.name: e
    for e in (
        Experiment(
            "S-1",
            "the model is already right: equal spread, logistic regression",
            (1.0,),
            1.0,
            _plain(lambda k: synthetic.Biomarkers(d=D, kappa=k)),
            lambda level: {"model": "linear"},
        ),
        Experiment(
            "S-2",
            "disease makes measurements more variable; a linear risk score",
            (1.0, 2.0, 4.0, 8.0),
            4.0,
            _plain(lambda k: synthetic.Biomarkers(d=D, kappa=k)),
            lambda level: {"model": "linear"},
        ),
        Experiment(
            "S-3",
            "look-alike negatives decide the threshold",
            (0.0, 0.01, 0.02, 0.05),
            0.02,
            _plain(lambda share: synthetic.LookAlikes(d=D, share=share)),
        ),
        Experiment(
            "S-4",
            "a rare subtype of the positive class",
            (0.05, 0.1, 0.2),
            0.1,
            _plain(lambda share: synthetic.RareSubtype(d=D, share=share)),
        ),
        Experiment(
            "S-5",
            "mislabelled positives (labels mined from reports)",
            (0.0, 0.05, 0.1, 0.2),
            0.0,
            _noisy,
        ),
        Experiment(
            "S-6",
            "a large network trained for a long time",
            tuple(
                (m, h, n)
                for n in (2_000, 20_000)
                for m, h in (("linear", ()), ("mlp", (32,)), ("mlp", (256, 256)))
            ),
            ("mlp", (32,), 20_000),
            _capacity,
            lambda level: {
                "model": level[0],
                "hidden": level[1],
                "weight_decay": 0.0,
                "epochs": 500,
                "checkpoints": LONG,
            },
        ),
        Experiment(
            "S-7",
            "rare events: few positives, or low prevalence",
            ((200, 0.1), (200, 0.02), (200, 0.005), (50, 0.02), (1000, 0.02)),
            (200, 0.02),
            _rare,
        ),
        Experiment(
            "S-8",
            "deployment drift: more look-alikes after training",
            (0.02, 0.05, 0.1),
            0.02,
            _drift,
        ),
    )
}


def level_tag(level: Any) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", str(level)).strip("_") or "level"


def _spec(exp: Experiment, level: Any, arm: str, params: dict[str, Any], seed: int,
          epochs: int | None) -> RunSpec:  # fmt: skip
    fields: dict[str, Any] = {
        "model": "mlp", "hidden": (32,), "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 1024,
        "epochs": 200, "checkpoints": CHECKPOINTS, "target": TARGET,
    }  # fmt: skip
    fields.update(exp.spec(level))
    if epochs is not None:  # smoke runs
        fields["epochs"] = epochs
        fields["checkpoints"] = tuple(c for c in fields["checkpoints"] if c <= epochs) or (epochs,)
    return RunSpec(task=f"{exp.name}/{level_tag(level)}", arm=arm, params=params, seed=seed,
                   **fields)  # fmt: skip


def _job(
    args: tuple[str, Any, int, list[tuple[str, dict[str, Any]]], float, int | None, str],
) -> int:
    name, level, seed, arms, scale, epochs, out = args
    torch.set_num_threads(1)
    exp = EXPERIMENTS[name]
    task = exp.task(level, seed, scale)
    for arm, params in arms:
        spec = _spec(exp, level, arm, params, seed, epochs)
        if "warmup_fraction" in params:  # I3: CE for this share of the training steps
            steps = -(-len(task.y_train) // spec.batch_size) * spec.epochs
            resolved = {k: v for k, v in params.items() if k != "warmup_fraction"}
            resolved["warmup_steps"] = max(1, round(params["warmup_fraction"] * steps))
            spec = _spec(exp, level, arm, resolved, seed, epochs)
        runner.run(task, spec, Path(out))
    return len(arms)


def _rows(folder: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for p in folder.rglob("runs.jsonl") for line in p.read_text().splitlines()
    ]


def choose(rows: Iterable[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Configuration of ``arm`` with the highest mean (over seeds) best-checkpoint val. sens."""
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        spec = row["spec"]
        if spec["arm"] != arm or row["status"] != "ok":
            continue
        checkpoints = set(spec["checkpoints"])
        best = max(h["val_sens"] for h in row["history"] if h["epoch"] in checkpoints)
        scores[json.dumps(spec["params"], sort_keys=True)].append(best)
    key = max(scores, key=lambda k: float(np.mean(scores[k])))
    return json.loads(key)  # type: ignore[no-any-return]


def _pool(jobs: list[Any], workers: int) -> None:
    if workers <= 1:
        for job in jobs:
            _job(job)
        return
    with get_context("spawn").Pool(workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"  {done}/{len(jobs)} jobs", flush=True)


def run_experiment(
    name: str,
    out: Path = OUT,
    seeds: Iterable[int] = SEEDS,
    tune_seeds: Iterable[int] = TUNE_SEEDS,
    scale: float = 1.0,
    epochs: int | None = None,
    workers: int = 10,
) -> None:
    exp = EXPERIMENTS[name]
    root = out / name
    tune = [(arm, params) for arm, grid in TUNED.items() for params in grid]
    jobs = [(name, exp.centre, s, tune, scale, epochs, str(root / "tune" / f"seed{s}"))
            for s in tune_seeds]  # fmt: skip
    print(f"{name}: tuning, {len(jobs)} jobs", flush=True)
    _pool(jobs, workers)
    tuned = _rows(root / "tune")
    chosen = {arm: choose(tuned, arm) for arm in TUNED}
    (root / "chosen.json").write_text(json.dumps(chosen, indent=2))
    arms = list(expected_params(chosen).items())
    jobs = [(name, level, s, arms, scale, epochs, str(root / level_tag(level) / f"seed{s}"))
            for level in exp.levels for s in seeds]  # fmt: skip
    print(f"{name}: evaluation, {len(jobs)} jobs, chosen {chosen}", flush=True)
    _pool(jobs, workers)


def expected_params(chosen: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parameters of every evaluated arm: CE, the tuned arms, and the controls, which take
    AUCReshaping's chosen setting."""
    return {"ce": {}, **chosen, **dict.fromkeys(CONTROLS, chosen["aucreshape"])}


def current_rows(root: Path, chosen: str = "chosen.json") -> list[dict[str, Any]]:
    """Evaluation rows (not tuning, not interventions) whose parameters match ``chosen``
    (``chosen.json``, or ``chosen_original.json`` for the setting before a grid extension)."""
    expected = expected_params(json.loads((root / chosen).read_text()))
    rows = []
    for path in sorted(root.rglob("runs.jsonl")):
        if "tune" in path.parts or "interventions" in path.relative_to(root).parts:
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["spec"]["params"] == expected[row["spec"]["arm"]]:
                row["_dir"] = str(path.parent)
                rows.append(row)
    return rows


def intervention_rows(root: Path) -> list[dict[str, Any]]:
    """Evaluation rows of the five interventions (I1-I5); one setting per arm was evaluated."""
    rows = []
    for path in sorted((root / "interventions").rglob("runs.jsonl")):
        if "tune" in path.relative_to(root / "interventions").parts:
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["_dir"] = str(path.parent)
            rows.append(row)
    return rows


def extend_experiment(
    name: str,
    out: Path = OUT,
    seeds: Iterable[int] = SEEDS,
    tune_seeds: Iterable[int] = TUNE_SEEDS,
    scale: float = 1.0,
    epochs: int | None = None,
    workers: int = 10,
) -> dict[str, dict[str, Any]]:
    """Re-tune the arms that picked the largest value of their grid, with ``EXTRA``."""
    exp = EXPERIMENTS[name]
    root = out / name
    chosen = json.loads((root / "chosen.json").read_text())
    edge = [arm for arm in TUNED if chosen[arm] == TUNED[arm][-1]]
    if not edge:
        print(f"{name}: no arm at a grid edge", flush=True)
        return chosen  # type: ignore[no-any-return]
    original = root / "chosen_original.json"
    if not original.exists():
        original.write_text(json.dumps(chosen, indent=2))
    tune = [(arm, params) for arm in edge for params in EXTRA[arm]]
    jobs = [(name, exp.centre, s, tune, scale, epochs, str(root / "tune" / f"seed{s}"))
            for s in tune_seeds]  # fmt: skip
    print(f"{name}: extending {edge}, {len(jobs)} jobs", flush=True)
    _pool(jobs, workers)
    tuned = _rows(root / "tune")
    new = {arm: choose(tuned, arm) for arm in TUNED}
    (root / "chosen.json").write_text(json.dumps(new, indent=2))
    changed = [arm for arm in TUNED if new[arm] != chosen[arm]]
    arms = [(arm, new[arm]) for arm in changed]
    if "aucreshape" in changed:
        arms += [(c, new["aucreshape"]) for c in CONTROLS]
    print(f"{name}: chosen {new}; re-evaluating {[a for a, _ in arms]}", flush=True)
    if arms:
        jobs = [(name, level, s, arms, scale, epochs, str(root / level_tag(level) / f"seed{s}"))
                for level in exp.levels for s in seeds]  # fmt: skip
        _pool(jobs, workers)
    return new  # type: ignore[no-any-return]


INTERVENTIONS = ("aucreshape_pw", "warm_aucreshape", "near_miss", "hard_negatives", "smoothed")
INTERVENTION_GRIDS = {
    "near_miss": [{"boost": b} for b in (0.05, 0.1, 0.2, 0.4)],
    "hard_negatives": [{"boost": b} for b in (1.0, 2.0, 4.0, 8.0)],
    "smoothed": [{"momentum": m} for m in (0.5, 0.8, 0.9, 0.95)],  # with the chosen boost
}


def intervene_experiment(
    name: str,
    out: Path = OUT,
    seeds: Iterable[int] = SEEDS,
    tune_seeds: Iterable[int] = TUNE_SEEDS,
    scale: float = 1.0,
    epochs: int | None = None,
    workers: int = 10,
) -> dict[str, dict[str, Any]]:
    """The five pre-registered interventions, at the current choices."""
    exp = EXPERIMENTS[name]
    root = out / name
    folder = root / "interventions"
    chosen = json.loads((root / "chosen.json").read_text())
    grids = {arm: list(grid) for arm, grid in INTERVENTION_GRIDS.items()}
    grids["smoothed"] = [{**chosen["aucreshape"], **m} for m in INTERVENTION_GRIDS["smoothed"]]
    tune = [(arm, params) for arm, grid in grids.items() for params in grid]
    jobs = [(name, exp.centre, s, tune, scale, epochs, str(folder / "tune" / f"seed{s}"))
            for s in tune_seeds]  # fmt: skip
    print(f"{name}: tuning interventions, {len(jobs)} jobs", flush=True)
    _pool(jobs, workers)
    tuned = _rows(folder / "tune")
    picked = {arm: choose(tuned, arm) for arm in grids}
    (root / "interventions_chosen.json").write_text(json.dumps(picked, indent=2))
    arms = [
        (
            "aucreshape_pw",
            {**chosen["aucreshape"], "pos_weight": chosen["weighted_bce"]["pos_weight"]},
        ),
        ("warm_aucreshape", {**chosen["aucreshape"], "warmup_fraction": 0.5}),
        *picked.items(),
    ]
    jobs = [(name, level, s, arms, scale, epochs, str(folder / level_tag(level) / f"seed{s}"))
            for level in exp.levels for s in seeds]  # fmt: skip
    print(f"{name}: evaluating interventions, {len(jobs)} jobs, {picked}", flush=True)
    _pool(jobs, workers)
    return picked  # type: ignore[no-any-return]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", choices=sorted(EXPERIMENTS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--extend", action="store_true", help="re-tune arms at a grid edge")
    parser.add_argument("--interventions", action="store_true", help="run I1-I5")
    args = parser.parse_args()
    for name in args.experiments:
        if args.extend:
            extend_experiment(name, args.out, workers=args.workers)
        elif args.interventions:
            intervene_experiment(name, args.out, workers=args.workers)
        else:
            run_experiment(name, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
