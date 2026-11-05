"""The older studies: first chest X-ray study, early benchmark, first credit-card run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.review import reanalysis_older as older
from benchmarks.review.stats import PairedGroup


def _groups(gain: float, seeds: int = 2) -> list[PairedGroup]:
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.3).astype(np.float32)
    noise = rng.normal(size=(seeds, 3000)).astype(np.float32)
    return [PairedGroup(y, noise + y, noise + (1 + gain) * y)]


def test_sampling_only_bootstrap_of_identical_arms_is_zero() -> None:
    estimate = older.sampling_only_bootstrap(_groups(0.0), n_boot=20, specificity=0.9)
    assert estimate["delta"] == 0.0
    assert estimate["lo"] == 0.0
    assert estimate["hi"] == 0.0


def test_sampling_only_bootstrap_reports_a_gain_in_points_per_seed() -> None:
    estimate = older.sampling_only_bootstrap(_groups(1.0), n_boot=20, specificity=0.9)
    assert estimate["delta"] > 5
    assert estimate["lo"] <= estimate["delta"] <= estimate["hi"]
    assert len(estimate["per_seed"]) == 2
    assert estimate["delta"] == pytest.approx(np.mean(estimate["per_seed"]))


def _fake_e4(root: Path) -> Path:
    """Disease logits; AUCReshaping's separate the classes better than the other arms'."""
    (root / "scores").mkdir(parents=True)
    rng = np.random.default_rng(0)
    y = (rng.random((3000, 15)) < 0.3).astype(np.uint8)
    y_val = (rng.random((800, 15)) < 0.3).astype(np.uint8)
    lines = []
    for seed in (0, 1):
        for arm, gain in (("ce", 0.0), ("ar_paper", 1.0), ("focal", 0.0), ("random_boost", 0.0)):
            key = f"nih_{arm}_{seed}"
            test = (rng.normal(size=y.shape) + (1 + gain) * y).astype(np.float16)
            val = (rng.normal(size=y_val.shape) + (1 + gain) * y_val).astype(np.float16)
            np.savez(
                root / "scores" / f"{key}.npz",
                test_val_sens=test,
                test_val_loss=test,
                test_last=test,
                test_5=test,
                val_5=val,
                y_test=y,
                y_val=y_val,
            )
            history = [
                {"epoch": 1, "val/effusion/sens@0.98": 0.1, "test/effusion/sens@0.98": 0.2 + gain},
                {"epoch": 2, "val/effusion/sens@0.98": 0.3, "test/effusion/sens@0.98": 0.4 + gain},
            ]
            lines.append(
                json.dumps(
                    {"key": key, "arm": arm, "seed": seed, "status": "ok", "history": history}
                )
            )
    (root / "runs.jsonl").write_text("\n".join(lines) + "\n")
    return root


def test_e4_contrasts_cover_the_design_and_hold_no_claim_from_two_seeds(tmp_path: Path) -> None:
    rows = older.e4_contrasts(_fake_e4(tmp_path), n_boot=5)
    keys = {(r["task"], r["convention"], r["rule"], r["arm"]) for r in rows}
    assert len(keys) == len(rows) == 3 * 2 * 3 * 3
    assert {r["arm"] for r in rows} == {"ce", "focal", "random_boost"}
    assert {r["interval"] for r in rows} == {"test sampling only, by image; 2 seeds"}
    assert {r["claim"] for r in rows} == {"none: 2 seeds give no interval over seeds"}


@pytest.mark.parametrize("convention", ["normal = 1 (paper)", "disease = 1"])
def test_e4_better_separation_is_a_gain_in_both_conventions(
    tmp_path: Path, convention: str
) -> None:
    rows = older.e4_contrasts(_fake_e4(tmp_path), n_boot=5)
    wanted = [r for r in rows if r["convention"] == convention and r["arm"] == "random_boost"]
    assert all(r["delta"] > 0 for r in wanted)


def test_e4_selection_on_effusion_alone_reads_the_history(tmp_path: Path) -> None:
    rows = older.e4_task_selected(_fake_e4(tmp_path))
    row = next(r for r in rows if r["arm"] == "random_boost")
    assert row["per_seed_epoch"] == [2, 2]
    assert row["per_seed"] == [pytest.approx(40.0), pytest.approx(40.0)]
    gain = next(r for r in rows if r["arm"] == "ar_paper")
    assert gain["per_seed"] == [pytest.approx(140.0), pytest.approx(140.0)]


def test_e4_validation_threshold_reports_realised_specificity(tmp_path: Path) -> None:
    rows = older.e4_validation_threshold(_fake_e4(tmp_path), epochs=(5,))
    assert {r["epoch"] for r in rows} == {5}
    assert {r["convention"] for r in rows} == {"normal = 1 (paper)", "disease = 1"}
    for row in rows:
        assert 90.0 < row["realised_specificity"] <= 100.0
        assert 0.0 <= row["sens_at_val_threshold"] <= 100.0


def _fake_raw(root: Path, name: str = "toy") -> Path:
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.5).astype(np.float32)
    torch.save(torch.from_numpy(y), root / f"{name}_y_test.pt")
    gains = {"aucreshape_prob": 1.0, "aucreshape_logit": 0.0, "aucreshape_weight": 0.0}
    for method in ("bce", "weighted_bce", *gains):
        for seed in range(100, 105):
            scores = rng.normal(size=3000) + (1 + gains.get(method, 0.0)) * y
            torch.save(
                torch.from_numpy(scores.astype(np.float32)), root / f"{name}_{method}_{seed}.pt"
            )
    return root


def test_s09_contrasts_keep_modes_apart_and_use_two_levels(tmp_path: Path) -> None:
    rows = older.s09_contrasts(_fake_raw(tmp_path / "raw"), "toy", target=0.9, n_boot=20)
    assert {(r["setting"], r["arm"]) for r in rows} == {
        (mode, arm) for mode in ("prob", "logit", "weight") for arm in ("bce", "weighted_bce")
    }
    assert {r["interval"] for r in rows} == {"two-level"}
    prob = next(r for r in rows if r["setting"] == "prob" and r["arm"] == "weighted_bce")
    assert prob["delta"] > 5
    assert prob["claim"] == "gain of at least about 1 point"
    assert len(prob["per_seed"]) == 5


def test_s11_contrast_reads_fraud_logits_in_the_paper_direction(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    fraud = (rng.random(4000) < 0.2).astype(np.float32)
    for seed in range(5):
        for arm, gain in (("ce", 0.0), ("ar_paper_cc", 1.0)):
            for rule in ("last", "val_loss", "val_sens"):
                z = rng.normal(size=4000) + (1 + gain) * fraud
                path = raw / f"creditcard_0_mlp_{arm}_{seed}_{rule}.pt"
                torch.save(torch.from_numpy(z.astype(np.float32)), path)
    rows = older.s11_contrasts(raw, fraud, n_boot=20, target=0.9)
    assert {r["rule"] for r in rows} == {"last", "val_loss", "val_sens"}
    assert all(r["delta"] > 0 for r in rows)
    assert all(r["convention"] == "normal = 1 (paper)" for r in rows)


def test_floors_list_each_studys_baseline() -> None:
    values = {("toy", "a"): [10.0, 12.0, 13.0], ("toy", "b"): [5.0, 5.5]}
    rows = older.floors(values)
    assert [r["pairs"] for r in rows] == [3, 1]
    assert rows[0]["study"] == "toy"
    assert rows[1]["max_abs"] == pytest.approx(0.5)
