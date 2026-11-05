"""Tests for the E4 additions to benchmarks.replication : NIH loading from the preprocessing
preprocessing, the focal and random-boost arms, run tracking, and the Kaggle plan."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from aucreshape import AUCReshapingLoss
from benchmarks import replication
from benchmarks.kaggle import collect, plans
from benchmarks.review import nih

COLUMNS = [*nih.FINDINGS, "No Finding"]
PATTERNS = [(), ("Effusion",), ("Pneumothorax",), ("Effusion", "Pneumothorax"), ("Atelectasis",)]


def fake_nih256(root: Path, n: int = 200, side: int = 8, blank: int | None = None) -> Path:
    """A tiny copy of the preprocessing output: images.npy (N, side, side) and labels.npz."""
    folder = root / "aucreshape-s12-nih256" / "nih256"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    images = rng.integers(1, 256, size=(n, side, side), dtype=np.uint8)
    if blank is not None:
        images[blank] = 0
    np.save(folder / "images.npy", images)
    labels = np.zeros((n, len(COLUMNS)), dtype=np.uint8)
    for i in range(n):
        found = PATTERNS[i % len(PATTERNS)] or ("No Finding",)
        for name in found:
            labels[i, COLUMNS.index(name)] = 1
    patient = np.arange(n) // 2
    split = np.where(patient % 4 == 0, "test", "train_val")
    np.savez(
        folder / "labels.npz",
        image=np.array([f"{i:08d}_000.png" for i in range(n)]),
        patient=patient,
        split=split,
        columns=np.array(COLUMNS),
        labels=labels,
    )
    return root


def test_load_nih_downsamples_and_keeps_the_official_split(tmp_path: Path) -> None:
    root = fake_nih256(tmp_path)
    data = replication.load_nih(root, size=4)
    meta = np.load(next(root.rglob("labels.npz")))
    full = np.load(next(root.rglob("images.npy")))
    sizes = {part: len(x) for part, x in data.x.items()}
    assert sum(sizes.values()) == 200
    assert sizes["test"] == int((meta["split"] == "test").sum())
    assert data.x["train"].dtype == torch.uint8
    assert data.x["train"].shape[1:] == (4, 4)
    # 2 x 2 box average, rounded, of the first test image
    first_test = int(np.flatnonzero(meta["split"] == "test")[0])
    expected = full[first_test].reshape(4, 2, 4, 2).astype(np.float64).mean((1, 3))
    np.testing.assert_allclose(data.x["test"][0].numpy(), np.round(expected), atol=0.5)
    # 15 labels: the 14 findings in NIH order, then "any finding"
    assert data.y["train"].shape[1] == 15
    assert data.tasks == {"effusion": 2, "pneumothorax": 7, "abnormal": 14}
    y = torch.cat([data.y[p] for p in ("train", "val", "test")])
    assert torch.equal(y[:, 14], y[:, :14].amax(1))


def test_load_nih_validation_split_is_patient_wise(tmp_path: Path) -> None:
    root = fake_nih256(tmp_path)
    data = replication.load_nih(root, size=4)
    patients = {p: set(data.patients[p].tolist()) for p in ("train", "val", "test")}
    assert not patients["train"] & patients["val"]
    assert not patients["train"] & patients["test"]
    assert not patients["val"] & patients["test"]
    train_val = len(patients["train"]) + len(patients["val"])
    assert len(patients["val"]) == round(0.1 * train_val)
    again = replication.load_nih(root, size=4)  # fixed split seed
    assert torch.equal(again.patients["val"], data.patients["val"])


def test_load_nih_rejects_blank_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="blank"):
        replication.load_nih(fake_nih256(tmp_path, blank=17), size=4)


def test_load_nih_rejects_a_size_that_does_not_divide(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divide"):
        replication.load_nih(fake_nih256(tmp_path), size=3)


PW = torch.tensor([0.5, 2.0, 1.0])


def batch(seed: int, n: int = 256, c: int = 3) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, c, generator=g) < 0.5).float()
    z = torch.randn(n, c, generator=g) + 1.5 * y
    return z, y


def test_focal_with_gamma_zero_equals_the_class_weighted_ce_arm() -> None:
    focal = replication.FocalBCE(0.0, PW)
    ce = AUCReshapingLoss(0.98, boost=0.0, num_labels=3, pos_weight=PW)
    z, y = batch(0)
    torch.testing.assert_close(focal(z, y), ce(z, y))


def test_focal_matches_its_definition_and_is_finite() -> None:
    focal = replication.FocalBCE(2.0, PW)
    z, y = batch(1)
    p = torch.sigmoid(z)
    by_hand = -(PW * y * (1 - p) ** 2 * torch.log(p) + (1 - y) * p**2 * torch.log(1 - p))
    z.requires_grad_(True)
    torch.testing.assert_close(focal(z, y), by_hand.mean())
    big = torch.tensor([[-1e4, 1e4, 0.0]], requires_grad=True)
    loss = focal(big, torch.tensor([[1.0, 0.0, 1.0]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert big.grad is not None
    assert torch.isfinite(big.grad).all()
    assert big.grad.abs().sum() > 0


def test_focal_without_gradients_is_weighted_bce() -> None:
    # Validation loss means the same thing in every arm (AUCReshapingLoss does the same).
    z, y = batch(2)
    with torch.no_grad():
        got = replication.FocalBCE(2.0, PW)(z, y)
    torch.testing.assert_close(got, F.binary_cross_entropy_with_logits(z, y, pos_weight=PW))


def paper_base(c: int = 3) -> AUCReshapingLoss:
    table = torch.zeros(c, 4)
    table[0] = torch.tensor([0.4, 0.2, 0.1, 0.1])
    table[2] = torch.tensor([0.4, 0.2, 0.1, 0.1])
    return AUCReshapingLoss(
        [0.90, 0.92, 0.95, 0.98], boost=table, num_labels=c, pos_weight=PW, window_size=1024
    )


def test_random_boost_without_shuffle_reproduces_aucreshaping() -> None:
    base = paper_base()
    control = replication.RandomBoost(copy.deepcopy(base), seed=0, shuffle=False)
    for step in range(12):
        z, y = batch(10 + step)
        z_a, z_b = z.clone().requires_grad_(True), z.clone().requires_grad_(True)
        loss_a, loss_b = base(z_a, y), control(z_b, y)
        torch.testing.assert_close(loss_b, loss_a)
        loss_a.backward()
        loss_b.backward()
        torch.testing.assert_close(z_b.grad, z_a.grad)
        torch.testing.assert_close(
            control.last_stats["boosted_fraction"], base.last_stats["boosted_fraction"]
        )
    assert float(base.last_stats["boosted_fraction"][0]) > 0  # the check is not vacuous


def test_random_boost_moves_boosts_within_each_label() -> None:
    base = paper_base()
    control = replication.RandomBoost(copy.deepcopy(base), seed=0)
    for step in range(8):  # same tracker state in both
        z, y = batch(30 + step)
        base(z.clone().requires_grad_(True), y).backward()
        control(z.clone().requires_grad_(True), y).backward()
    z, y = batch(99)
    base(z, y)
    control(z, y)
    fraction = control.last_stats["boosted_fraction"]
    torch.testing.assert_close(fraction, base.last_stats["boosted_fraction"])
    assert float(fraction[1]) == 0.0  # label without boosts stays unboosted
    assert not torch.equal(control.last_boosted, base_boosted(base, z, y))


def base_boosted(base: AUCReshapingLoss, z: Tensor, y: Tensor) -> Tensor:
    from aucreshape import boost_values

    b = boost_values(z, y, base.tracker.active_thresholds(), base.boost_table, combine="band")
    return (b > 0) & (y > 0.5)


def test_random_boost_control_is_reproducible() -> None:
    losses = []
    for _ in range(2):
        control = replication.RandomBoost(paper_base(), seed=3)
        out = []
        for step in range(6):
            z, y = batch(50 + step)
            loss = control(z.requires_grad_(True), y)
            loss.backward()
            out.append(loss.detach())
        losses.append(torch.stack(out))
    torch.testing.assert_close(losses[0], losses[1])


class Tiny(nn.Module):
    def __init__(self, arch: str, num_outputs: int) -> None:
        super().__init__()
        self.fc = nn.Linear(16, num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x.float().flatten(1) / 255)  # type: ignore[no-any-return]


def run_main(monkeypatch: pytest.MonkeyPatch, data: Path, out: Path, *extra: str) -> None:
    monkeypatch.setattr(replication, "ImageNet", Tiny)
    monkeypatch.setenv("AUCRESHAPE_GIT_SHA", "a" * 40)
    argv = ["replication", "nih", "--size", "4", "--epochs", "2", "--seeds", "0",
            "--arms", "ce,ar_paper,focal,random_boost", "--data", str(data), "--out", str(out),
            *extra]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    replication.main()


def test_nih_runs_write_tracked_rows_and_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = fake_nih256(tmp_path / "input")
    out = tmp_path / "kernel" / "results" / "e4" / "all"
    run_main(monkeypatch, data, out)
    rows = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in rows] == ["ce", "ar_paper", "focal", "random_boost"]
    for r in rows:
        assert r["status"] == "ok"
        assert r["git_sha"] == "a" * 40
        assert r["key"] == f"nih_4_swav_resnet50_{r['arm']}_0_2"
        assert len(r["history"]) == 2
        h = r["history"][-1]
        assert "val/effusion/sens@0.98" in h
        assert "test/abnormal/auc" in h
        assert np.isfinite(h["train_loss"])
        if r["arm"] != "focal":
            assert len(h["boosted_fraction"]) == 15
        scores = np.load(out / "scores" / f"{r['key']}.npz")
        assert scores["test_2"].dtype == np.float16
        assert scores["test_2"].shape == scores["y_test"].shape
        assert {"val_2", "y_val", "test_val_loss", "test_val_sens", "test_last"} <= set(scores)
    # the collector understands the output, and a second call skips finished runs
    summary = collect.merge(tmp_path / "kernel", tmp_path / "merged", kernel="me/k")
    assert summary["new"] == 4
    assert summary["failed"] == 0
    run_main(monkeypatch, data, out)
    assert len((out / "runs.jsonl").read_text().splitlines()) == 4


def test_a_run_past_the_deadline_stops_and_keeps_its_partial_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AUCRESHAPE_DEADLINE", "0")
    out = tmp_path / "out"
    run_main(monkeypatch, fake_nih256(tmp_path / "input"), out)
    rows = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    for r in rows:
        assert r["status"] == "stopped: deadline after epoch 1"
        assert len(r["history"]) == 1
        assert "test_1" in np.load(out / "scores" / f"{r['key']}.npz")


def test_e4_settings_are_the_registered_ones() -> None:
    job = replication.JOBS["nih"]
    assert job["epochs"] == 200
    assert job["eval_every"] == 1
    assert job["arch"] == "swav_resnet50"
    assert replication.ARMS["focal"] == {"focal": 2.0}
    assert replication.ARMS["random_boost"]["control"] == "random"
    assert replication.CHECKPOINTS == (5, 10, 20, 50, 100, 150, 200)


def test_s12e_plan_runs_every_arm_and_seed_once() -> None:
    kernels = plans.PLANS["s12e"]
    assert len(kernels) == 6
    runs = []
    for k in kernels:
        assert k.gpu
        assert k.workers == 2
        assert len(k.jobs) == 2
        assert k.kernel_sources == [plans.NIH256]
        for job in k.jobs:
            assert job[:2] == ["benchmarks.replication", "nih"]
            args = dict(zip(job[2::2], job[3::2], strict=True))
            assert args["--epochs"] == "200"
            assert args["--size"] == "128"
            assert args["--out"].startswith("/kaggle/working/results/e4/")
            runs.append((args["--arms"], args["--seeds"]))
    expected = {(a, str(s)) for a in ("ce", "ar_paper", "focal", "random_boost") for s in range(3)}
    assert sorted(runs) == sorted(expected)
    first = {
        (a, s)
        for k in kernels[:2]
        for a, s in [(j[j.index("--arms") + 1], j[j.index("--seeds") + 1]) for j in k.jobs]
    }
    assert first == {(a, "0") for a in ("ce", "ar_paper", "focal", "random_boost")}
