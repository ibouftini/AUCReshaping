"""Tests for benchmarks.review.analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aucreshape import metrics
from benchmarks.review import analysis


@pytest.mark.parametrize("specificity", [0.9, 0.95, 0.98, 0.999])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])  # float16 scores create ties
def test_fast_sensitivity_matches_the_library(specificity: float, dtype: type) -> None:
    rng = np.random.default_rng(0)
    for n in (50, 1_000, 20_000):
        y = (rng.random(n) < 0.2).astype(np.float32)
        s = (rng.normal(size=n) + y).astype(dtype)
        expected = float(
            metrics.sensitivity_at_specificity(
                torch.from_numpy(s.astype(np.float32)), torch.from_numpy(y), specificity
            )
        )
        # The library returns float32 (~1e-8 rounding); one sample off would be >= 5e-5.
        assert analysis.sens_at_spec(s, y, specificity) == pytest.approx(expected, abs=1e-6)


def test_selected_checkpoint_is_the_best_validation_sensitivity() -> None:
    row = {
        "spec": {"checkpoints": [10, 20, 50]},
        "history": [
            {"epoch": e, "val_sens": v} for e, v in ((10, 0.3), (15, 0.9), (20, 0.5), (50, 0.4))
        ],
    }
    assert analysis.selected_epoch(row) == 20  # epoch 15 is not a checkpoint
    assert analysis.selected_epoch(row, rule="last") == 50


def test_paired_effect_from_runs(tmp_path: Path) -> None:
    # Two seeds, each with its own test set; arm "b" scores every positive higher.
    rows = []
    for seed in (0, 1):
        for arm, shift in (("a", 0.0), ("b", 5.0)):
            key = f"{arm}{seed}"
            rows.append({"key": key, "status": "ok",
                         "spec": {"arm": arm, "seed": seed, "checkpoints": [1]},
                         "history": [{"epoch": 1, "val_sens": 0.5}]})  # fmt: skip
            rng = np.random.default_rng(seed)
            y = (rng.random(2000) < 0.3).astype(np.float32)
            s = rng.normal(size=2000) + shift * y
            (tmp_path / "scores").mkdir(exist_ok=True)
            np.savez(tmp_path / "scores" / f"{key}.npz", test_1=s.astype(np.float16))
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    labels = {seed: (np.random.default_rng(seed).random(2000) < 0.3).astype(np.float32)
              for seed in (0, 1)}  # fmt: skip
    est = analysis.paired_effect(rows, tmp_path, labels.__getitem__, "a", "b", 0.98, n_boot=200)
    assert est.point > 0.5  # a fraction: +0.5 is +50 points of sensitivity
    assert est.cells == 2
