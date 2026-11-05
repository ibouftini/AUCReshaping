"""One tiny run of each evaluation pipeline on generated data.

    python -m benchmarks.review.smoke

It checks that each pipeline runs from end to end in seconds on CPU, and says nothing about results.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from benchmarks import replication
from benchmarks.review import cc_reconcile, reanalysis, runner


def _split(gen: torch.Generator, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = (torch.rand(n, generator=gen) < 0.3).float()
    return torch.randn(n, 4, generator=gen) + y[:, None], y


def synthetic_runner(out: Path) -> None:
    """Run the synthetic laboratory's runner on every kind of arm."""
    gen = torch.Generator().manual_seed(0)
    (xt, yt), (xv, yv), (xs, ys) = _split(gen, 600), _split(gen, 300), _split(gen, 300)
    task = runner.Task("smoke", xt, yt, xv, yv, xs, ys)
    base = runner.RunSpec(
        task="smoke", arm="ce", params={}, seed=0, hidden=(8,), dropout=0.1, batch_size=64,
        epochs=2, checkpoints=(2,), target=0.9,
    )  # fmt: skip
    boost = {"boost": 0.3, "min_negatives": 20}
    arms = {
        "ce": {},
        "aucreshape": boost,
        "weighted_bce": {"pos_weight": 4.0},
        "focal": {"gamma": 2.0},
        "random_boost": boost,
    }
    for arm, params in arms.items():
        row = runner.run(task, replace(base, arm=arm, params=dict(params)), out)
        if row is None or row["status"] != "ok":
            raise RuntimeError(f"{arm}: {row}")


def real_data_pipeline(_: Path) -> None:
    """Run the real-data training loop, tabular form, on every arm; images need weights."""
    gen = torch.Generator().manual_seed(0)
    (xt, yt), (xv, yv), (xs, ys) = _split(gen, 1200), _split(gen, 300), _split(gen, 300)
    data = replication.Data(
        {"train": xt, "val": xv, "test": xs}, {"train": yt, "val": yv, "test": ys}, {"fraud": 0},
        image=False,
    )  # fmt: skip
    job = replication.Job(
        name="smoke", size=0, arch="mlp", epochs=2, lr=0.01, batch_size=64, optimizer="sgd",
        pos_weight=torch.tensor(5.0), window_size=1024, eval_every=1,
    )  # fmt: skip
    for arm in ("ce", "ar_paper", "focal", "random_boost"):
        result = replication.train_one(job, data, arm, 0, "cpu")
        if result["status"] != "ok":
            raise RuntimeError(f"{arm}: {result['status']}")


def credit_card_baseline(_: Path) -> None:
    """Run the credit-card reconciliation's training."""
    gen = torch.Generator().manual_seed(0)
    x, y = _split(gen, 600)
    x_test, _labels = _split(gen, 200)
    config = replace(cc_reconcile.PAPER, epochs=2, batch_size=64, widths=(8,), checkpoints=(2,))
    result = cc_reconcile.train_run(x, y, x_test, config, seed=0)
    if result.status != "ok":
        raise RuntimeError(result.status)


def direct_contrasts(out: Path) -> None:
    """Run the direct contrasts, from saved scores through to one estimate."""
    rng = np.random.default_rng(0)
    name, tag = "S-0", "0.0"
    scenario = out / name
    folder = scenario / tag / "seed0"
    (folder / "scores").mkdir(parents=True)
    chosen = {"aucreshape": {"boost": 0.4}, "weighted_bce": {"pos_weight": 8.0}}
    (scenario / "chosen.json").write_text(json.dumps(chosen))
    y = (rng.random(1000) < 0.3).astype(np.float32)
    lines = []
    for arm, params in (("aucreshape", {"boost": 0.4}), ("random_boost", {"boost": 0.4})):
        scores = (rng.normal(size=1000) + y).astype(np.float16)
        np.savez(folder / "scores" / f"{arm}.npz", test_2=scores)
        spec = {
            "task": f"{name}/{tag}",
            "arm": arm,
            "params": params,
            "seed": 0,
            "checkpoints": [2],
        }
        history = [{"epoch": 2, "val_loss": 0.1, "val_sens": 0.5}]
        lines.append(json.dumps({"key": arm, "status": "ok", "spec": spec, "history": history}))
    (folder / "runs.jsonl").write_text("\n".join(lines) + "\n")
    index = reanalysis.build_index(out, ("random_boost",))
    key = f"{name}|{tag}|0|test"
    labels = {key: reanalysis.pack_labels(y)}
    row = reanalysis.estimate(
        reanalysis.Contrast(name, tag, "prob", "random_boost"), index, out, labels, {key: 1000},
        n_boot=5,
    )  # fmt: skip
    if not np.isfinite(row["ar_sens"]):
        raise RuntimeError(row)


PIPELINES: dict[str, Callable[[Path], None]] = {
    "synthetic runner": synthetic_runner,
    "real-data training loop": real_data_pipeline,
    "credit-card baseline": credit_card_baseline,
    "direct contrasts": direct_contrasts,
}


def run(pipelines: Mapping[str, Callable[[Path], None]], root: Path) -> list[dict[str, str]]:
    """Run every pipeline in its own folder; a failure does not stop the others."""
    report = []
    for i, (name, pipeline) in enumerate(pipelines.items()):
        folder = root / f"pipeline-{i}"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            pipeline(folder)
        except Exception:  # every pipeline is reported
            report.append({"pipeline": name, "status": "failed", "detail": traceback.format_exc()})
            continue
        report.append({"pipeline": name, "status": "ok", "detail": ""})
    return report


def main() -> None:
    """Run every pipeline and print what happened to each."""
    with tempfile.TemporaryDirectory() as folder:
        report = run(PIPELINES, Path(folder))
    for entry in report:
        print(f"{entry['status']:7s} {entry['pipeline']}\n{entry['detail']}".rstrip())
    sys.exit(int(any(entry["status"] != "ok" for entry in report)))


if __name__ == "__main__":
    main()
