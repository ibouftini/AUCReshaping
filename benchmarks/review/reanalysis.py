"""Direct contrasts from saved results, without training (exploratory).

The earlier tables report differences against CE, so the claim "weighted BCE and the random
control do as well" was only inferred from overlapping intervals. This module reports the direct
paired contrasts on the synthetic laboratory: AUCReshaping minus each control and minus weighted
BCE, with the two-level interval and the +/-1 point region. ``prob`` (the paper's Eq. 1)
and ``weight`` stay in separate rows.

The bootstrap over 330 contrasts takes hours of CPU, so it runs on private Kaggle CPU kernels:

    python -m benchmarks.review.reanalysis bundle --out DIR      # index, labels and code
    python -m benchmarks.review.reanalysis kernels --user U --out DIR --shards 5
    python -m benchmarks.review.reanalysis local --out DIR       # validation threshold, noise floor
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.review import analysis, stats

TARGET = 0.98
REGION = 1.0  # points: the region of practical equivalence
TUNED = ("aucreshape", "weighted_bce")
SEEDS = tuple(range(10))
BUNDLE_MODULES = ("__init__.py", "analysis.py", "stats.py", "reanalysis.py")


def claim_label(lo: float, hi: float, region: float = REGION) -> str:
    """Return the wording that a 95% interval, in points, permits."""
    if lo > hi:
        raise ValueError(f"lo must not exceed hi; got {lo} and {hi}.")
    if lo > region:
        return "gain of at least about 1 point"
    if hi < -region:
        return "loss of at least about 1 point"
    if -region <= lo and hi <= region:
        return "no difference larger than 1 point"
    if lo > 0:
        return "gain, size uncertain"
    if hi < 0:
        return "loss, size uncertain"
    return "inconclusive"


def read_settings(scenario: Path) -> dict[str, dict[str, Any]]:
    """Return each AUCReshaping setting evaluated in one scenario, keyed by its mode.

    ``chosen.json`` holds the final setting, and ``chosen_original.json``, where it exists, the
    setting before the grid extension. A setting without ``mode`` uses the default, ``prob``.
    """
    settings: dict[str, dict[str, Any]] = {}
    for file in ("chosen.json", "chosen_original.json"):
        path = scenario / file
        if not path.exists():
            continue
        chosen = json.loads(path.read_text())
        mode = chosen["aucreshape"].get("mode", "prob")
        if mode in settings:
            raise ValueError(f"Two {mode} settings in {scenario}.")
        settings[mode] = chosen
    return settings


def build_index(root: Path, controls: Sequence[str]) -> dict[str, Any]:
    """Map scenario, setting, level, arm and seed to a run's score file and selected epochs.

    Parameters
    ----------
    root : Path
        The folder of the synthetic results, holding one folder per scenario.
    controls : sequence of str
        Arms that take AUCReshaping's setting.

    Returns
    -------
    dict
        ``index[name][setting][tag][arm][str(seed)]`` holds ``key``, ``folder`` (relative to
        ``root``), ``status`` and the checkpoints selected by ``val``, ``loss`` and ``last``.
    """
    index: dict[str, Any] = {}
    for scenario in sorted(p for p in root.iterdir() if (p / "chosen.json").exists()):
        settings = read_settings(scenario)
        tree: dict[str, Any] = {setting: {} for setting in settings}
        expected = {
            setting: {
                **{arm: chosen[arm] for arm in TUNED},
                **dict.fromkeys(controls, chosen["aucreshape"]),
            }
            for setting, chosen in settings.items()
        }
        for path in sorted(scenario.rglob("runs.jsonl")):
            parts = path.relative_to(scenario).parts
            if "tune" in parts or "interventions" in parts:
                continue
            for line in path.read_text().splitlines():
                row = json.loads(line)
                spec = row["spec"]
                tag = spec["task"].split("/", 1)[1]
                ok = row["status"] == "ok"
                run = {
                    "key": row["key"],
                    "folder": str(path.parent.relative_to(root)),
                    "status": row["status"],
                    "val": analysis.selected_epoch(row, "val") if ok else None,
                    "loss": analysis.selected_epoch(row, "loss") if ok else None,
                    "last": analysis.selected_epoch(row, "last") if ok else None,
                }
                for setting, want in expected.items():
                    if spec["arm"] == "ce" or want.get(spec["arm"]) == spec["params"]:
                        arms = tree[setting].setdefault(tag, {})
                        arms.setdefault(spec["arm"], {})[str(spec["seed"])] = run
        index[scenario.name] = tree
    return index


@dataclass(frozen=True)
class Contrast:
    """AUCReshaping minus ``arm``, in one scenario, level and setting."""

    name: str
    tag: str
    setting: str
    arm: str


def contrasts(index: dict[str, Any]) -> list[Contrast]:
    """Return every contrast of AUCReshaping with a control or with weighted BCE."""
    plan = []
    for name, tree in index.items():
        for setting, levels in tree.items():
            for tag, arms in levels.items():
                plan += [
                    Contrast(name, tag, setting, arm)
                    for arm in sorted(arms)
                    if arm not in ("aucreshape", "ce", "focal")
                ]
    return plan


def shard(plan: list[Contrast], index: int, shards: int) -> list[Contrast]:
    """Return the contrasts of one kernel."""
    return plan[index::shards]


def pack_labels(y: np.ndarray) -> np.ndarray:
    """Store labels, shape (N,), as bits."""
    return np.packbits(y > 0.5)


def unpack_labels(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`pack_labels`; returns booleans, shape (N,).

    ``np.unpackbits`` pads with zeros when asked for too many bits, which would silently corrupt
    labels, so a length that does not fit the stored bytes is an error.
    """
    if (n + 7) // 8 != len(packed):
        raise ValueError(f"{len(packed)} bytes of labels cannot hold {n} labels.")
    return np.unpackbits(packed, count=n).astype(bool)


def _scores(root: Path, run: dict[str, Any], split: str, epoch: int) -> np.ndarray:
    with np.load(root / run["folder"] / "scores" / f"{run['key']}.npz") as saved:
        return saved[f"{split}_{epoch}"].astype(np.float32)


def _points(scores: np.ndarray, y: np.ndarray) -> float:
    return analysis.sens_at_spec(scores, y, TARGET)


def estimate(
    contrast: Contrast,
    index: dict[str, Any],
    root: Path,
    labels: Mapping[str, np.ndarray],
    meta: dict[str, int],
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Estimate one contrast at the checkpoint of highest validation sensitivity.

    Returns
    -------
    dict
        Row with ``delta``, ``lo`` and ``hi`` in points (AUCReshaping minus the arm), the mean
        sensitivity of each arm, the per-seed differences, their SD and the permitted wording.
    """
    arms = index[contrast.name][contrast.setting][contrast.tag]
    ours, theirs = arms["aucreshape"], arms[contrast.arm]
    seeds = sorted(set(ours) & set(theirs), key=int)
    groups = []
    for seed in seeds:
        key = f"{contrast.name}|{contrast.tag}|{seed}|test"
        y = unpack_labels(labels[key], meta[key]).astype(np.float32)
        a = _scores(root, theirs[seed], "test", theirs[seed]["val"])
        b = _scores(root, ours[seed], "test", ours[seed]["val"])
        groups.append(stats.PairedGroup(y, a[None], b[None]))
    result = stats.two_level_bootstrap(groups, _points, n_boot=n_boot)
    ar = [100 * _points(g.b[0], g.y) for g in groups]
    control = [100 * _points(g.a[0], g.y) for g in groups]
    per_seed = [a - c for a, c in zip(ar, control, strict=True)]
    lo, hi = 100 * result.lo, 100 * result.hi
    return {
        **asdict(contrast),
        "seeds": len(seeds),
        "ar_sens": float(np.mean(ar)),
        "control_sens": float(np.mean(control)),
        "delta": 100 * result.point,
        "lo": lo,
        "hi": hi,
        "p_value": result.p_value,
        "gains": result.seeds_with_gain,
        "paired_sd": float(np.std(per_seed, ddof=1)),
        "per_seed": per_seed,
        "claim": claim_label(lo, hi),
    }


def threshold_cut(val: np.ndarray, y_val: np.ndarray, specificity: float = TARGET) -> float:
    """Return the score above which a case counts as positive, fixed on validation negatives."""
    negative = np.sort(val[y_val < 0.5])
    return float(negative[int(np.ceil(specificity * len(negative))) - 1])


def validation_threshold(
    val: np.ndarray,
    y_val: np.ndarray,
    test: np.ndarray,
    y_test: np.ndarray,
    specificity: float = TARGET,
) -> tuple[float, float]:
    """Sensitivity and realised specificity on test at a threshold fixed on validation.

    Parameters
    ----------
    val, y_val : ndarray, shape (V,)
        Validation scores and labels.
    test, y_test : ndarray, shape (T,)
        Test scores and labels.
    specificity : float, default 0.98
        The specificity asked of the validation negatives.

    Returns
    -------
    sensitivity, specificity : float
        A case counts as positive when its score is above the threshold.
    """
    positive = test > threshold_cut(val, y_val, specificity)
    return float(positive[y_test > 0.5].mean()), float((~positive)[y_test < 0.5].mean())


def noise_floor(values: Sequence[float]) -> dict[str, float]:
    """Summarise the differences between every pair of repeats of one arm, in the input's unit."""
    gaps = [abs(a - b) for a, b in combinations(values, 2)]
    return {
        "pairs": len(gaps),
        "median_abs": float(np.median(gaps)),
        "p95_abs": float(np.quantile(gaps, 0.95)),
        "max_abs": float(np.max(gaps)),
    }


def local_tables(
    root: Path, index: dict[str, Any], labels: Mapping[str, np.ndarray], meta: dict[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validation threshold and noise floor: one row per arm, and one per level for CE."""
    thresholds, floors = [], []
    for name, tree in index.items():
        for setting, levels in tree.items():
            for tag, arms in levels.items():
                for arm, runs in arms.items():
                    if arm == "ce" and setting != next(iter(tree)):
                        continue  # the same CE runs appear under every setting
                    sens, spec, roc = [], [], []
                    for seed, run in sorted(runs.items(), key=lambda item: int(item[0])):
                        base = f"{name}|{tag}|{seed}"
                        y_test = unpack_labels(labels[f"{base}|test"], meta[f"{base}|test"])
                        y_val = unpack_labels(labels[f"{base}|val"], meta[f"{base}|val"])
                        test = _scores(root, run, "test", run["val"])
                        val = _scores(root, run, "val", run["val"])
                        at_threshold = validation_threshold(
                            val, y_val.astype(np.float32), test, y_test.astype(np.float32)
                        )
                        sens.append(100 * at_threshold[0])
                        spec.append(100 * at_threshold[1])
                        roc.append(100 * _points(test, y_test.astype(np.float32)))
                    thresholds.append(
                        {
                            "name": name,
                            "tag": tag,
                            "setting": "" if arm == "ce" else setting,
                            "arm": arm,
                            "sens_at_val_threshold": float(np.mean(sens)),
                            "realised_specificity": float(np.mean(spec)),
                            "realised_specificity_sd": float(np.std(spec, ddof=1)),
                            "sens_from_test_roc": float(np.mean(roc)),
                        }
                    )
                    if arm == "ce":
                        floors.append({"name": name, "tag": tag, **noise_floor(roc)})
    return thresholds, floors


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows with the union of their fields."""
    names = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(rows)


# Shared with the forked workers.
_STATE: dict[str, Any] = {}


def _work(contrast: Contrast) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        row = estimate(contrast, _STATE["index"], _STATE["root"], _STATE["labels"], _STATE["meta"])
        row["status"] = "ok"
    except Exception as error:  # every contrast counts: a failure is a result
        row = {**asdict(contrast), "status": "error", "error": repr(error)}
    row["seconds"] = time.perf_counter() - start
    return row


def _find_results() -> Path:
    found = glob.glob("/kaggle/input/**/e1/S-1/chosen.json", recursive=True)
    if found:
        return Path(found[0]).parents[1]
    archive = glob.glob("/kaggle/input/**/aucreshape-s12-e1-eval.tar", recursive=True)
    if not archive:
        raise FileNotFoundError("No synthetic results in /kaggle/input.")
    target = Path("/tmp/aucreshape_results")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", archive[0], "-C", str(target)], check=True)
    return target / "e1"


def run_kernel(index: int, shards: int, workers: int = 4) -> None:
    """Estimate this kernel's contrasts and append them to ``contrasts.jsonl``."""
    bundle = Path(glob.glob("/kaggle/input/**/index.json", recursive=True)[0]).parent
    out = Path("/kaggle/working/results")
    out.mkdir(parents=True, exist_ok=True)
    tree = json.loads((bundle / "index.json").read_text())
    _STATE.update(
        index=tree,
        root=_find_results(),
        labels=dict(np.load(bundle / "labels.npz")),
        meta=json.loads((bundle / "meta.json").read_text()),
    )
    plan = shard(contrasts(tree), index, shards)
    context = multiprocessing.get_context("fork")
    with context.Pool(workers) as pool, (out / "contrasts.jsonl").open("a") as log:
        for row in pool.imap_unordered(_work, plan):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row["name"], row["tag"], row["setting"], row["arm"], row["status"], flush=True)


def make_kernel(root: Path, user: str, index: int, shards: int, datasets: Sequence[str]) -> Path:
    """Write one private, offline CPU kernel; return its folder."""
    slug = f"aucreshape-s13a-contrasts-{index}"
    folder = root / slug
    folder.mkdir(parents=True, exist_ok=True)
    modules = ", ".join(repr(m) for m in BUNDLE_MODULES)
    (folder / "kernel.py").write_text(
        "import glob\nimport shutil\nimport sys\nfrom pathlib import Path\n\n"
        "# A Kaggle dataset is flat, so the package is rebuilt from the bundle's files.\n"
        "bundle = Path(glob.glob('/kaggle/input/**/index.json', recursive=True)[0]).parent\n"
        "code = Path('/tmp/aucreshape_code')\n"
        "(code / 'benchmarks' / 'review').mkdir(parents=True, exist_ok=True)\n"
        "(code / 'benchmarks' / '__init__.py').write_text('')\n"
        f"for module in ({modules}):\n"
        "    shutil.copy(bundle / module, code / 'benchmarks' / 'review' / module)\n"
        "sys.path.insert(0, str(code))\n"
        "from benchmarks.review import reanalysis  # noqa: E402\n\n"
        f"reanalysis.run_kernel({index}, {shards})\n"
    )
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": list(datasets),
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


def build_labels(index: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Regenerate the validation and test labels of every run from the task generators."""
    from benchmarks.review import e1

    labels: dict[str, np.ndarray] = {}
    sizes: dict[str, int] = {}
    for name in index:
        experiment = e1.EXPERIMENTS[name]
        for level in experiment.levels:
            tag = e1.level_tag(level)
            for seed in SEEDS:
                task = experiment.task(level, seed, 1.0)
                for split, y in (("test", task.y_test), ("val", task.y_val)):
                    key = f"{name}|{tag}|{seed}|{split}"
                    labels[key] = pack_labels(y.numpy())
                    sizes[key] = len(y)
    return labels, sizes


def make_bundle(out: Path, root: Path) -> Path:
    """Write the index, the labels and this code as the input folder of a private dataset."""
    from benchmarks.review import e1

    out.mkdir(parents=True, exist_ok=True)
    index = build_index(root, e1.CONTROLS)
    labels, meta = build_labels(index)
    (out / "index.json").write_text(json.dumps(index))
    (out / "meta.json").write_text(json.dumps(meta))
    np.savez_compressed(out / "labels.npz", **labels)
    code = Path(__file__).resolve().parent
    for module in BUNDLE_MODULES:  # flat: the Kaggle CLI skips sub-folders of a dataset
        shutil.copy2(code / module, out / module)
    return out


def main() -> None:
    """Run one of the local commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("--out", type=Path, required=True)
    bundle.add_argument("--root", type=Path, default=Path("benchmarks/results/s12/e1"))
    kernels = commands.add_parser("kernels")
    kernels.add_argument("--user", required=True)
    kernels.add_argument("--out", type=Path, required=True)
    kernels.add_argument("--shards", type=int, default=5)
    kernels.add_argument("--dataset", action="append", required=True)
    local = commands.add_parser("local")
    local.add_argument("--bundle", type=Path, required=True)
    local.add_argument("--out", type=Path, required=True)
    local.add_argument("--root", type=Path, default=Path("benchmarks/results/s12/e1"))
    args = parser.parse_args()
    if args.command == "bundle":
        print(make_bundle(args.out, args.root))
    elif args.command == "kernels":
        for index in range(args.shards):
            print(make_kernel(args.out, args.user, index, args.shards, args.dataset))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        tree = json.loads((args.bundle / "index.json").read_text())
        labels = dict(np.load(args.bundle / "labels.npz"))
        meta = json.loads((args.bundle / "meta.json").read_text())
        thresholds, floors = local_tables(args.root, tree, labels, meta)
        write_csv(thresholds, args.out / "validation_threshold.csv")
        write_csv(floors, args.out / "noise_floor_e1.csv")


if __name__ == "__main__":
    main()
