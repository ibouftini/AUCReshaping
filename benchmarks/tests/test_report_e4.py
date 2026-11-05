"""Tests for benchmarks.review.report_e4 (chest X-ray summary) on a tiny fake result set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.review import report_e4

TASKS = {"effusion": 2, "pneumothorax": 7, "abnormal": 14}


def fake_e4(root: Path, same: bool = False) -> Path:
    """Two arms x two seeds; ``same`` gives the AUCReshaping arm CE's logits."""
    (root / "scores").mkdir(parents=True)
    rng = np.random.default_rng(0)
    y = (rng.random((400, 15)) < 0.3).astype(np.uint8)
    rows = []
    for seed in (0, 1):
        ce = (rng.normal(size=(400, 15)) + 1.5 * y).astype(np.float16)
        for arm in ("ce", "ar_paper"):
            key = f"nih_128_swav_resnet50_{arm}_{seed}_4"
            logits = (
                ce if (arm == "ce" or same) else (ce + rng.normal(size=ce.shape)).astype(np.float16)
            )
            np.savez(root / "scores" / f"{key}.npz", test_val_loss=logits, test_last=logits,
                     test_val_sens=logits, y_test=y)  # fmt: skip
            history = [{"epoch": e, **{f"test/{t}/sens@0.98": 0.1 * e for t in TASKS},
                        "boosted_fraction": [0.5 / e] * 15 if arm != "ce" else [0.0] * 15}
                       for e in range(1, 5)]  # fmt: skip
            rows.append({"key": key, "status": "ok", "arm": arm, "seed": seed,
                         "selected": {"val_loss": {"epoch": 2}}, "history": history})  # fmt: skip
    (root / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_summary_covers_tasks_conventions_rules_and_arms(tmp_path: Path) -> None:
    table = report_e4.summary(fake_e4(tmp_path))
    keys = {(r["task"], r["convention"], r["rule"], r["arm"]) for r in table}
    assert len(keys) == len(table)
    assert {r["task"] for r in table} == set(TASKS)
    assert {r["convention"] for r in table} == {"normal = 1 (paper)", "disease = 1"}
    assert {r["rule"] for r in table} == {
        "highest validation sensitivity",
        "lowest validation loss",
        "epoch 200",
    }
    for r in table:
        assert len(r["per_seed"]) == 2
        assert all(0.0 <= v <= 100.0 for v in r["per_seed"])
        if r["arm"] == "ce":
            assert r["delta"] is None
        else:
            assert r["delta"] == pytest.approx(np.mean(r["per_seed_delta"]))


def test_identical_logits_give_zero_delta(tmp_path: Path) -> None:
    table = report_e4.summary(fake_e4(tmp_path, same=True))
    deltas = [r["delta"] for r in table if r["arm"] != "ce"]
    assert deltas
    assert all(d == pytest.approx(0.0) for d in deltas)


def test_curves_average_the_seeds(tmp_path: Path) -> None:
    rows = report_e4.load(fake_e4(tmp_path))
    epochs, mean = report_e4.curve(rows, "ar_paper", "test/effusion/sens@0.98")
    assert list(epochs) == [1, 2, 3, 4]
    np.testing.assert_allclose(mean, [10.0, 20.0, 30.0, 40.0])
    _, boosted = report_e4.curve(rows, "ar_paper", "boosted_fraction", column=2)
    np.testing.assert_allclose(boosted, [50.0, 25.0, 50.0 / 3, 12.5])
