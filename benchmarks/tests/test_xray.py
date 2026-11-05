"""The new chest X-ray pipeline: a single-output model fine-tuned from pretrained weights."""

from __future__ import annotations

import json
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from aucreshape import AUCReshapingLoss
from benchmarks.review import xray

COLUMNS = ["Atelectasis", "Effusion", "No Finding"]


def _fake_root(
    root: Path, n: int = 240, side: int = 12, with_test: bool = True, separation: int = 120
) -> Path:
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    disease = (rng.random(n) < 0.3).astype(np.uint8)
    images = (rng.integers(0, 80, (n, side, side)) + separation * disease[:, None, None]).astype(
        np.uint8
    )
    np.save(root / "images_trainval.npy", images)
    labels = np.stack([np.zeros(n, dtype=np.uint8), disease, 1 - disease], axis=1)
    np.savez(
        root / "labels_trainval.npz",
        image=np.array([f"{i}.png" for i in range(n)]),
        patient=np.arange(n) // 2,
        columns=np.array(COLUMNS),
        labels=labels,
    )
    if with_test:
        np.save(root / "images_test.npy", images[:60])
        np.savez(
            root / "labels_test.npz",
            image=np.array([f"t{i}.png" for i in range(60)]),
            patient=np.arange(60) + 10_000,
            columns=np.array(COLUMNS),
            labels=labels[:60],
        )
    return root


def _tiny(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(144, 8), nn.ReLU(), nn.Linear(8, 1))


def _run(arm: str = "ce", **kw: object) -> xray.Run:
    fields = {"arm": arm, "seed": 3, "lr": 1e-3, "epochs": 3, "batch_size": 32, "weights": "toy"}
    return xray.Run(**(fields | kw))  # type: ignore[arg-type]


def test_patient_split_keeps_each_patient_on_one_side() -> None:
    patients = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 20) + np.repeat(np.arange(20) * 10, 10)
    train, val = xray.patient_split(patients, fraction=0.1, split_seed=7)
    assert set(patients[train]).isdisjoint(patients[val])
    assert sorted([*train, *val]) == list(range(len(patients)))
    assert 0.05 < len(val) / len(patients) < 0.2
    again = xray.patient_split(patients, fraction=0.1, split_seed=7)
    assert np.array_equal(val, again[1])
    assert not np.array_equal(val, xray.patient_split(patients, 0.1, split_seed=8)[1])


def test_pilot_data_never_needs_the_test_files(tmp_path: Path) -> None:
    root = _fake_root(tmp_path / "nih", with_test=False)
    data = xray.load_data(root, "Effusion", include_test=False)
    assert data.x_test is None
    assert data.y_test is None
    assert len(data.x_train) + len(data.x_val) == 240


def test_test_data_loads_only_when_asked(tmp_path: Path) -> None:
    root = _fake_root(tmp_path / "nih")
    data = xray.load_data(root, "Effusion", include_test=True)
    assert data.x_test is not None
    assert data.y_test is not None
    assert len(data.x_test) == 60
    assert set(data.patients_test.tolist()).isdisjoint(data.patients_train.tolist())
    with pytest.raises(FileNotFoundError):
        xray.load_data(
            _fake_root(tmp_path / "other", with_test=False), "Effusion", include_test=True
        )


def test_labels_are_the_requested_task_with_disease_as_one(tmp_path: Path) -> None:
    data = xray.load_data(_fake_root(tmp_path / "nih"), "Effusion", include_test=False)
    assert set(data.y_train.tolist()) == {0.0, 1.0}
    assert data.y_train.dtype == torch.float32


def _batch(seed: int = 0, n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, generator=gen) < 0.7).float()
    return torch.randn(n, generator=gen) + 1.5 * y, y


def _base(boost: float = 0.3, **kw: object) -> AUCReshapingLoss:
    return AUCReshapingLoss(0.9, boost=boost, window_size=1024, min_negatives=10, **kw)  # type: ignore[arg-type]


def _ready(loss: nn.Module) -> None:
    for step in range(8):
        z, y = _batch(step + 100)
        z.requires_grad_(True)
        loss(z, y).backward()


def _extra_gradient(
    loss: nn.Module, z: torch.Tensor, y: torch.Tensor
) -> tuple[float, torch.Tensor]:
    z = z.clone().requires_grad_(True)
    loss(z, y).sum().backward()
    assert z.grad is not None
    base = nn.functional.binary_cross_entropy_with_logits(z.detach(), y, reduction="none")
    assert base is not None
    positive = y > 0.5
    magnitude = z.grad[positive].abs() * len(y)  # undo the mean
    plain = 1.0 - torch.sigmoid(z.detach()[positive])
    return float((magnitude - plain).sum()), magnitude


@pytest.mark.parametrize("rule", ["random", "above"])
@pytest.mark.parametrize(("mode", "boost"), [("prob", 0.3), ("weight", 4.0)])
def test_matched_control_delivers_the_same_extra_gradient(
    rule: str, mode: str, boost: float
) -> None:
    target = _base(boost, mode=mode)
    control = xray.MatchedBoost(_base(boost, mode=mode), rule=rule, seed=0)
    _ready(target)
    _ready(control)
    z, y = _batch(1)
    ours, _ = _extra_gradient(target, z, y)
    theirs, _ = _extra_gradient(control, z, y)
    assert ours > 0
    assert theirs == pytest.approx(ours, rel=1e-3)


def test_random_control_boosts_as_many_positives_as_are_flagged() -> None:
    control = xray.MatchedBoost(_base(), rule="random", seed=0)
    _ready(control)
    z, y = _batch(1)
    control(z.clone().requires_grad_(True), y)
    assert int(control.last_boosted.sum()) == int(control.last_flagged.sum())
    assert not torch.equal(control.last_boosted, control.last_flagged)
    assert bool((y[control.last_boosted] > 0.5).all())
    assert 0.0 < control.last_overlap < 1.0


def test_above_control_boosts_only_positives_the_threshold_would_not_flag() -> None:
    control = xray.MatchedBoost(_base(), rule="above", seed=0)
    _ready(control)
    z, y = _batch(1)
    control(z.clone().requires_grad_(True), y)
    assert control.last_boosted.any()
    assert not (control.last_boosted & control.last_flagged).any()
    assert control.last_overlap == 0.0
    assert bool((y[control.last_boosted] > 0.5).all())


def test_matched_control_is_plain_bce_before_thresholds_exist() -> None:
    control = xray.MatchedBoost(_base(), rule="random", seed=0)
    z, y = _batch(1, n=8)
    z.requires_grad_(True)
    loss = control(z, y)
    assert float(loss) == pytest.approx(float(nn.functional.binary_cross_entropy_with_logits(z, y)))


def test_matched_control_stays_finite_when_recipients_are_confident() -> None:
    control = xray.MatchedBoost(_base(0.5), rule="above", seed=0)
    _ready(control)
    z, y = _batch(1)
    z = z + 60.0 * y  # positives far above the threshold have almost no gradient of their own
    z.requires_grad_(True)
    loss = control(z, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_matched_control_does_not_touch_the_global_random_stream() -> None:
    control = xray.MatchedBoost(_base(), rule="random", seed=0)
    _ready(control)
    torch.manual_seed(5)
    expected = torch.rand(1)
    torch.manual_seed(5)
    z, y = _batch(1)
    control(z.clone().requires_grad_(True), y)
    assert torch.equal(torch.rand(1), expected)


def test_control_reduces_to_validation_bce_without_gradients() -> None:
    control = xray.MatchedBoost(_base(), rule="random", seed=0)
    _ready(control)
    z, y = _batch(1)
    with torch.no_grad():
        loss = control(z, y)
    assert float(loss) == pytest.approx(float(nn.functional.binary_cross_entropy_with_logits(z, y)))


def test_every_arm_can_be_built() -> None:
    weight = torch.tensor(0.5)
    for arm, params in xray.ARM_EXAMPLES.items():
        loss = xray.make_loss(arm, params, weight, window_size=256, seed=0)
        z, y = _batch(2)
        out = loss(z.clone().requires_grad_(True), y)
        assert torch.isfinite(out)


def test_unknown_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="arm"):
        xray.make_loss("magic", {}, torch.tensor(1.0), window_size=256, seed=0)


def _train(tmp_path: Path, run: xray.Run, name: str = "nih", **kw: object) -> dict:
    root = tmp_path / name
    if not root.exists():
        _fake_root(root)
    data = xray.load_data(root, "Effusion", include_test=run.evaluate_test)
    return xray.train_run(data, run, build_model=_tiny, device="cpu", out=tmp_path / "out", **kw)  # type: ignore[arg-type]


def test_a_run_records_what_the_protocol_needs(tmp_path: Path) -> None:
    row = _train(tmp_path, _run("aucreshape", params={"boost": 0.2, "min_negatives": 8}))
    assert row["status"] == "ok"
    assert len(row["history"]) == 3
    first = row["history"][0]
    for field in ("train_loss", "val_loss", "val_sens", "val_auc", "boosted_fraction", "seconds"):
        assert field in first
    assert {"val_sens", "val_loss", "last"} == set(row["selected"])
    assert first["recipient_overlap"] is None  # targeted: its recipients are the flagged set
    assert row["images_per_second"] > 0
    with np.load(tmp_path / "out" / f"{row['key']}.npz") as saved:
        assert saved["val_3"].shape == (24,)
        assert saved["val_3"].dtype == np.float32
        assert not [key for key in saved.files if key.startswith("test")]


def test_arms_share_initialization_and_batch_order(tmp_path: Path) -> None:
    # A boost of zero makes AUCReshaping the baseline, so a difference could only come from them.
    ce = _train(tmp_path, _run("ce"))
    zero = _train(tmp_path, _run("aucreshape", params={"boost": 0.0, "min_negatives": 8}))
    ce_scores = np.load(tmp_path / "out" / f"{ce['key']}.npz")["val_3"]
    zero_scores = np.load(tmp_path / "out" / f"{zero['key']}.npz")["val_3"]
    # The tiny network's logits reach the hundreds, and the two losses differ by float rounding.
    np.testing.assert_allclose(zero_scores, ce_scores, rtol=1e-4)


def test_seeds_change_the_run(tmp_path: Path) -> None:
    one = _train(tmp_path, _run("ce", seed=1))
    two = _train(tmp_path, _run("ce", seed=2))
    assert one["history"][0]["train_loss"] != two["history"][0]["train_loss"]


def test_test_scores_are_saved_only_for_confirmatory_runs(tmp_path: Path) -> None:
    row = _train(tmp_path, _run("ce", evaluate_test=True, seed=9))
    with np.load(tmp_path / "out" / f"{row['key']}.npz") as saved:
        assert saved["test_3"].shape == (60,)
    # Test metrics never enter the history that selection reads.
    assert not [key for key in row["history"][0] if "test" in key]


def test_selection_does_not_depend_on_test_data(tmp_path: Path) -> None:
    first = _train(tmp_path, _run("ce", evaluate_test=True, seed=11))
    root = tmp_path / "nih"
    with np.load(root / "labels_test.npz") as saved:
        fields = dict(saved)
    fields["labels"] = 1 - fields["labels"]
    np.savez(root / "labels_test.npz", **fields)
    second = _train(tmp_path, _run("ce", evaluate_test=True, seed=11, tag="again"))
    assert first["selected"] == second["selected"]


def test_sampler_oversamples_disease_and_is_reproducible() -> None:
    y = torch.tensor([1.0] * 10 + [0.0] * 90)
    one = xray.epoch_order(y, sampler=True, generator=torch.Generator().manual_seed(0))
    two = xray.epoch_order(y, sampler=True, generator=torch.Generator().manual_seed(0))
    assert torch.equal(one, two)
    assert len(one) == 100
    assert 0.3 < float(y[one].mean()) < 0.7
    plain = xray.epoch_order(y, sampler=False, generator=torch.Generator().manual_seed(0))
    assert sorted(plain.tolist()) == list(range(100))


def test_direction_b_boosts_disease(tmp_path: Path) -> None:
    row = _train(tmp_path, _run("ce", boost_class="disease", seed=4))
    assert row["status"] == "ok"
    assert row["config"]["boost_class"] == "disease"
    with pytest.raises(ValueError, match="boost_class"):
        _train(tmp_path, _run("ce", boost_class="other", seed=5))


def test_a_diverging_run_is_recorded_not_raised(tmp_path: Path) -> None:
    row = _train(tmp_path, _run("ce", lr=1e30, seed=6))
    assert row["status"] != "ok"
    assert row["status"].startswith("failed")


def test_seed_ranges_stay_apart() -> None:
    assert set(xray.PILOT_SEEDS).isdisjoint(xray.TUNING_SEEDS)
    assert max(xray.TUNING_SEEDS) < 2000
    assert not {j.seed for j in xray.pilot_jobs("chex")} - set(xray.PILOT_SEEDS)
    assert not [j for j in xray.pilot_jobs("chex") if j.evaluate_test]


def test_gpu_kernel_is_private_and_offline(tmp_path: Path) -> None:
    jobs = xray.pilot_jobs("chex")
    folder = xray.make_kernel(
        tmp_path, "someone", "pilot-0", jobs, ("someone/code",), ("someone/prep",)
    )
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/aucreshape-s13c-pilot-0"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["dataset_sources"] == ["someone/code"]
    assert meta["kernel_sources"] == ["someone/prep"]
    compile((folder / meta["code_file"]).read_text(), "kernel", "exec")


def test_a_matched_control_logs_its_overlap_with_the_flagged_set(tmp_path: Path) -> None:
    # Classes that overlap, so that some positives fall below the threshold.
    _fake_root(tmp_path / "hard", separation=2)
    params = {"boost": 0.2, "target_specificity": 0.9, "min_negatives": 8}
    row = _train(tmp_path, _run("matched_random", params=params, seed=8), name="hard")
    assert any(h["recipient_overlap"] is not None for h in row["history"])
    assert any(h["boosted_fraction"] for h in row["history"])
    for record in row["history"]:
        if record["boosted_fraction"] is not None:
            assert 0.0 <= record["boosted_fraction"] <= 1.0


def test_code_bundle_holds_the_code_and_nothing_private(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for name in (
        "src/aucreshape/__init__.py",
        "benchmarks/__init__.py",
        "benchmarks/review/__init__.py",
        "benchmarks/review/xray.py",
        "benchmarks/review/publish.py",
        "benchmarks/tests/test_xray.py",
        "benchmarks/results/s13a/bundle/copy.py",
        "dev/state/PROGRESS.md",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n")
    folder = xray.make_bundle(source, tmp_path / "bundle", "someone")
    meta = json.loads((folder / "dataset-metadata.json").read_text())
    assert meta["id"] == "someone/aucreshape-s13c-code"
    with tarfile.open(folder / "code.tar") as archive:
        names = set(archive.getnames())
    assert {"src/aucreshape/__init__.py", "benchmarks/review/xray.py", "code_manifest.txt"} <= names
    assert not [n for n in names if "results" in n or "publish" in n or "tests" in n or "dev/" in n]


def test_kernel_records_which_code_ran(tmp_path: Path) -> None:
    folder = xray.make_kernel(tmp_path, "someone", "x", xray.pilot_jobs("chex"), ("a/b",), ("c/d",))
    assert "AUCRESHAPE_CODE" in (folder / "kernel.py").read_text()


def test_tuning_covers_every_setting_on_every_tuning_seed_plus_ce() -> None:
    jobs = xray.tuning_jobs("pc", 1e-4)
    assert len(jobs) == 3 * (1 + len(xray.AR_TUNING_GRID) + len(xray.BCE_TUNING_GRID))
    assert len({job.key for job in jobs}) == len(jobs)
    assert {job.seed for job in jobs} == set(xray.TUNING_SEEDS)
    assert not [job for job in jobs if job.evaluate_test]
    assert {job.arm for job in jobs} == {"ce", "aucreshape", "weighted_bce"}
    # Every arm shares the design: only the loss differs.
    assert {(job.lr, job.epochs, job.batch_size, job.sampler, job.weights) for job in jobs} == {
        (1e-4, 10, 64, False, "pc")
    }
    for seed in xray.TUNING_SEEDS:
        chosen = [job for job in jobs if job.seed == seed]
        assert sum(job.arm == "ce" for job in chosen) == 1
        assert [j.params for j in chosen if j.arm == "aucreshape"] == list(xray.AR_TUNING_GRID)
        assert [j.params["factor"] for j in chosen if j.arm == "weighted_bce"] == list(
            xray.BCE_TUNING_GRID
        )


def test_tuning_grids_have_equal_size_and_hold_the_setting_first_run() -> None:
    assert len(xray.AR_TUNING_GRID) == len(xray.BCE_TUNING_GRID)
    assert {"combine": "band", "min_ratio": 0.1} in xray.AR_TUNING_GRID
    assert {g["combine"] for g in xray.AR_TUNING_GRID} == {"band", "sum"}
    # Both span the same strengths, about 1.3x to 15x, as measured on the pilot's models.
    assert min(xray.BCE_TUNING_GRID) == pytest.approx(1.25)
    assert max(xray.BCE_TUNING_GRID) == pytest.approx(15.0)


def test_tuning_kernels_split_the_jobs_without_loss(tmp_path: Path) -> None:
    jobs = xray.tuning_jobs("pc", 1e-4)
    folders = xray.tuning_kernels(tmp_path, "someone", jobs, ("a/code",), ("a/prep",), parts=2)
    assert len(folders) == 2
    keys: list[str] = []
    for folder in folders:
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        assert meta["is_private"] is True
        assert meta["enable_internet"] is False
        text = (folder / "kernel.py").read_text()
        keys += [job.key for job in jobs if repr(job.params) in text and str(job.seed) in text]
    assert {job.key for job in jobs} <= set(keys)
    assert [json.loads((f / "kernel-metadata.json").read_text())["id"] for f in folders] == [
        "someone/aucreshape-s13c-tune-0",
        "someone/aucreshape-s13c-tune-1",
    ]


def _tuned_row(arm: str, seed: int, params: dict[str, object], sens: float) -> dict[str, object]:
    return {
        "key": f"{arm}_{seed}_{params}",
        "status": "ok",
        "config": {"arm": arm, "seed": seed, "params": params, "evaluate_test": False},
        "history": [
            {"epoch": 1, "val_sens": sens - 0.1, "val_loss": 0.1},
            {"epoch": 2, "val_sens": sens, "val_loss": 0.2},
        ],
        "selected": {"val_sens": 2, "val_loss": 1, "last": 2},
    }


def _tuned_rows(mean_for: dict[float, float]) -> list[dict[str, object]]:
    return [
        _tuned_row("weighted_bce", seed, {"factor": f}, sens)
        for f, sens in mean_for.items()
        for seed in xray.TUNING_SEEDS
    ]


def test_choice_takes_the_highest_mean_validation_sensitivity() -> None:
    rows = _tuned_rows({1.25: 0.30, 2.0: 0.33, 4.0: 0.31})
    out = xray.choose_settings(rows)["weighted_bce"]
    assert out["chosen"] == {"factor": 2.0}
    assert out["edge"] is None
    assert [round(s["mean_val_sens"], 2) for s in out["settings"]] == [0.30, 0.33, 0.31]


def test_choice_reports_a_grid_edge() -> None:
    assert (
        xray.choose_settings(_tuned_rows({1.25: 0.4, 2.0: 0.3, 4.0: 0.3}))["weighted_bce"]["edge"]
        == "gentle"
    )
    assert (
        xray.choose_settings(_tuned_rows({1.25: 0.3, 2.0: 0.3, 4.0: 0.4}))["weighted_bce"]["edge"]
        == "strong"
    )


def test_a_setting_with_a_failed_run_cannot_be_chosen_and_stays_visible() -> None:
    rows = _tuned_rows({1.25: 0.30, 2.0: 0.33})
    rows[3] = {**rows[3], "status": "failed: not finite at epoch 2"}
    out = xray.choose_settings(rows)["weighted_bce"]
    assert out["chosen"] == {"factor": 1.25}
    assert [s["complete"] for s in out["settings"]] == [True, False]


def test_choice_refuses_rows_that_read_test_data_or_other_seeds() -> None:
    rows = _tuned_rows({1.25: 0.30})
    leaked = {**rows[0], "config": {**rows[0]["config"], "evaluate_test": True}}  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="test"):
        xray.choose_settings([leaked])
    other = {**rows[0], "config": {**rows[0]["config"], "seed": 2000}}  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="seed"):
        xray.choose_settings([other])


def _coded(tmp_path: Path, arm: str = "aucreshape", seed: int = 2000) -> dict[str, object]:
    return _train(tmp_path, _run(arm, seed=seed, code="arm-abcd", evaluate_test=True))


def test_a_coded_run_hides_its_arm_from_the_open_record(tmp_path: Path) -> None:
    row = _coded(tmp_path)
    assert row["key"].endswith("arm-abcd_2000")  # type: ignore[attr-defined]
    text = json.dumps(row)
    assert "aucreshape" not in text
    assert "boosted_fraction" not in text
    assert "recipient_overlap" not in text
    assert "train_loss" not in text
    assert row["config"]["code"] == "arm-abcd"  # type: ignore[index]
    assert "arm" not in row["config"]  # type: ignore[operator]
    assert "params" not in row["config"]  # type: ignore[operator]


def test_a_coded_run_keeps_what_selection_needs_and_what_was_hidden(tmp_path: Path) -> None:
    row = _coded(tmp_path)
    assert row["selected"]["val_sens"] >= 1  # type: ignore[index]
    assert "val_sens" in row["history"][0]  # type: ignore[index]
    sealed = json.loads((tmp_path / "out" / "sealed" / f"{row['key']}.json").read_text())
    assert sealed["arm"] == "aucreshape"
    assert sealed["history"][0]["boosted_fraction"] is not None


def test_an_uncoded_run_stays_as_it_was(tmp_path: Path) -> None:
    row = _train(tmp_path, _run("aucreshape", seed=3))
    assert row["config"]["arm"] == "aucreshape"
    assert "boosted_fraction" in row["history"][0]
    assert not (tmp_path / "out" / "sealed").exists()


def test_codes_are_stable_for_a_salt_and_differ_between_salts() -> None:
    arms = ("ce", "aucreshape", "matched_random", "weighted_bce")
    one = xray.arm_codes(arms, "salt-1")
    assert one == xray.arm_codes(arms, "salt-1")
    assert one != xray.arm_codes(arms, "salt-2")
    assert len(set(one.values())) == 4
    for arm, code in one.items():
        assert arm not in code


def test_confirmatory_jobs_are_coded_shuffled_and_read_test_data() -> None:
    chosen = {
        "aucreshape": {"combine": "band", "min_ratio": 0.75},
        "weighted_bce": {"factor": 3.0},
    }
    seeds = range(2000, 2004)
    jobs = xray.confirmatory_jobs("pc", 1e-4, chosen, seeds, "salt")
    assert len(jobs) == 6 * 4
    assert all(job.evaluate_test and job.code for job in jobs)
    assert {job.seed for job in jobs} == set(seeds)
    assert min(job.seed for job in jobs) >= 2000
    assert {(j.lr, j.epochs, j.batch_size, j.sampler, j.weights) for j in jobs} == {
        (1e-4, 10, 64, False, "pc")
    }
    # The order hides the arms, and the same salt repeats it.
    assert [j.arm for j in jobs[:3]] != ["ce", "aucreshape", "weighted_bce"]
    assert jobs == xray.confirmatory_jobs("pc", 1e-4, chosen, seeds, "salt")


def test_confirmatory_jobs_refuse_seeds_outside_the_range() -> None:
    chosen = {"aucreshape": {}, "weighted_bce": {"factor": 3.0}}
    with pytest.raises(ValueError, match="2000"):
        xray.confirmatory_jobs("pc", 1e-4, chosen, [1999], "salt")


def test_the_job_list_does_not_land_in_the_kernel_output(tmp_path: Path) -> None:
    folder = xray.make_kernel(tmp_path, "someone", "x", xray.pilot_jobs("chex"), ("a/b",), ("c/d",))
    text = (folder / "kernel.py").read_text()
    assert "/kaggle/working/jobs.json" not in text


def test_a_failed_coded_run_does_not_reveal_its_arm(tmp_path: Path) -> None:
    job = _run("matched_random", code="arm-1", seed=2000)
    row = xray.failure_row(job, ValueError("matched_random broke"), tmp_path)
    assert row["status"] == "failed"
    assert "matched_random" not in json.dumps(row)
    sealed = json.loads((tmp_path / "sealed" / f"{row['key']}.json").read_text())
    assert sealed["arm"] == "matched_random"
    assert "broke" in sealed["error"]


def test_a_failed_open_run_keeps_its_error_in_the_record(tmp_path: Path) -> None:
    row = xray.failure_row(_run("ce"), ValueError("broke"), tmp_path)
    assert row["status"].startswith("failed")
    assert "broke" in row["status"]
    assert row["config"]["arm"] == "ce"


def test_the_grid_edge_does_not_depend_on_the_order_of_the_rows() -> None:
    rows = _tuned_rows({15.0: 0.3, 1.25: 0.4, 6.0: 0.3, 3.0: 0.3})
    out = xray.choose_settings(rows)["weighted_bce"]
    assert out["chosen"] == {"factor": 1.25}
    assert out["edge"] == "gentle"
    assert [s["params"]["factor"] for s in out["settings"]] == [1.25, 3.0, 6.0, 15.0]


@pytest.mark.parametrize(
    ("best", "edge"),
    [
        ({"combine": "band", "min_ratio": 0.75}, "gentle"),
        ({"combine": "band", "min_ratio": 0.25}, None),
        ({"combine": "sum", "min_ratio": 0.05}, "strong"),
    ],
)
def test_aucreshape_settings_run_from_gentle_to_strong(
    best: dict[str, object], edge: str | None
) -> None:
    rows = [
        _tuned_row("aucreshape", seed, dict(setting), 0.4 if setting == best else 0.3)
        for setting in reversed(xray.AR_TUNING_GRID)
        for seed in xray.TUNING_SEEDS
    ]
    out = xray.choose_settings(rows)["aucreshape"]
    assert out["chosen"] == best
    assert out["edge"] == edge
    assert [s["params"] for s in out["settings"]] == list(xray.AR_TUNING_GRID)


def test_a_wider_grid_keeps_its_gentle_and_strong_ends() -> None:
    wider = ({"combine": "band", "min_ratio": 0.9}, *xray.AR_TUNING_GRID)
    rows = [
        _tuned_row("aucreshape", seed, dict(setting), 0.4 if setting == wider[0] else 0.3)
        for setting in wider
        for seed in xray.TUNING_SEEDS
    ]
    assert xray.choose_settings(rows)["aucreshape"]["edge"] == "gentle"


def test_the_extension_adds_one_gentler_setting_to_each_grid_on_every_seed() -> None:
    jobs = xray.extension_jobs("pc", 1e-4)
    assert len(jobs) == 2 * len(xray.TUNING_SEEDS)
    assert {job.seed for job in jobs} == set(xray.TUNING_SEEDS)
    assert not [job for job in jobs if job.evaluate_test]
    assert {job.arm for job in jobs} == {"aucreshape", "weighted_bce"}
    for job in jobs:
        if job.arm == "aucreshape":
            assert job.params == {"combine": "band", "min_ratio": 0.9}
            assert job.params not in xray.AR_TUNING_GRID
        else:
            assert job.params == {"factor": 1.1}
            assert job.params["factor"] < min(xray.BCE_TUNING_GRID)
    known = {job.key for job in xray.tuning_jobs("pc", 1e-4)}
    assert not known & {job.key for job in jobs}


CHOSEN = {
    "aucreshape": {"combine": "band", "min_ratio": 0.75},
    "weighted_bce": {"factor": 3.0},
}


def test_the_six_arms_have_the_settings_of_the_design() -> None:
    jobs = xray.confirmatory_jobs("pc", 1e-4, CHOSEN, [2000], "salt")
    assert len({job.key for job in jobs}) == 6
    assert len({job.code for job in jobs}) == 6
    settings = {(job.arm, json.dumps(job.params, sort_keys=True)) for job in jobs}
    six = json.dumps(xray.CONTRAST_SETTING, sort_keys=True)
    assert settings == {
        ("ce", "{}"),
        ("aucreshape", json.dumps(CHOSEN["aucreshape"], sort_keys=True)),
        ("weighted_bce", json.dumps(CHOSEN["weighted_bce"], sort_keys=True)),
        ("aucreshape", six),
        ("matched_random", six),
        ("matched_above", six),
    }


def test_the_codes_for_analysis_match_the_codes_on_the_runs() -> None:
    jobs = xray.confirmatory_jobs("pc", 1e-4, CHOSEN, [2000], "salt")
    codes = xray.confirmatory_codes("salt")
    assert set(codes) == {
        "ce",
        "aucreshape",
        "weighted_bce",
        "aucreshape_6x",
        "matched_random_6x",
        "matched_above_6x",
    }
    assert {job.code for job in jobs} == set(codes.values())
    for job in jobs:
        name = next(n for n, c in codes.items() if c == job.code)
        assert name.removesuffix("_6x") == job.arm
        if name.endswith("_6x"):
            assert job.params == xray.CONTRAST_SETTING


def test_the_contrast_setting_is_the_one_the_first_study_ran() -> None:
    assert xray.CONTRAST_SETTING == {"combine": "band", "min_ratio": 0.1}
    assert xray.CONTRAST_SETTING in xray.AR_TUNING_GRID


def test_confirmatory_kernels_stay_short_enough_and_lose_no_run(tmp_path: Path) -> None:
    jobs = xray.confirmatory_jobs("pc", 1e-4, CHOSEN, range(2000, 2010), "salt")
    folders = xray.confirmatory_kernels(tmp_path, "someone", jobs, ("a/code",), ("a/prep",))
    assert len(folders) == 5
    total = 0
    for folder in folders:
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        assert meta["is_private"] is True
        assert meta["enable_internet"] is False
        assert meta["enable_gpu"] is True
        assert meta["id"].startswith("someone/aucreshape-s13c-confirm-")
        kernel_jobs = xray.read_kernel_jobs(folder)
        assert all(job["evaluate_test"] for job in kernel_jobs)
        assert len(kernel_jobs) <= xray.MAX_RUNS_PER_KERNEL
        total += len(kernel_jobs)
    assert total == len(jobs)


def test_the_salt_is_created_once_and_kept_private(tmp_path: Path) -> None:
    path = tmp_path / "private" / "salt.txt"
    first = xray.read_or_create_salt(path)
    assert len(first) >= 32
    assert path.stat().st_mode & 0o077 == 0
    assert xray.read_or_create_salt(path) == first


def test_a_coded_kernel_does_not_show_its_arms_in_the_source(tmp_path: Path) -> None:
    jobs = xray.confirmatory_jobs("pc", 1e-4, CHOSEN, [2000, 2001], "salt")
    folder = xray.make_kernel(tmp_path, "someone", "c", jobs, ("a/b",), ("c/d",))
    text = (folder / "kernel.py").read_text()
    jobs_line = next(line for line in text.splitlines() if line.startswith("JOBS = "))
    for word in ("aucreshape", "matched_random", "weighted_bce", "min_ratio", "factor", "arm"):
        assert word not in jobs_line.removeprefix("JOBS = json.loads(zlib.decompress(base64.b64")
    compile(text, "kernel", "exec")
    assert xray.read_kernel_jobs(folder) == [asdict(job) for job in jobs]


def test_an_open_kernel_still_shows_its_jobs(tmp_path: Path) -> None:
    jobs = xray.pilot_jobs("chex")
    folder = xray.make_kernel(tmp_path, "someone", "p", jobs, ("a/b",), ("c/d",))
    assert "'arm': 'ce'" in (folder / "kernel.py").read_text()
    assert xray.read_kernel_jobs(folder) == [asdict(job) for job in jobs]


def test_a_coded_run_hides_its_timing(tmp_path: Path) -> None:
    row = _coded(tmp_path)
    assert "seconds" not in json.dumps(row)
    assert "images_per_second" not in row
    sealed = json.loads((tmp_path / "out" / "sealed" / f"{row['key']}.json").read_text())
    assert sealed["seconds"] > 0
    assert sealed["images_per_second"] > 0
    assert "seconds" in sealed["history"][0]


def _masses(tmp_path: Path, arm: str) -> list[dict[str, object]]:
    params = {"boost": 0.2, "min_negatives": 8}
    return _train(tmp_path, _run(arm, params=params, seed=5))["history"]


def test_targeted_and_control_arms_log_the_extra_gradient(tmp_path: Path) -> None:
    for arm in ("aucreshape", "matched_random"):
        history = _masses(tmp_path, arm)
        assert all(h["extra_mass"] >= 0.0 for h in history)
        assert any(h["extra_mass"] > 0.0 for h in history)
    assert "extra_mass" in xray.HIDDEN_FIELDS
    assert "capped_share" in xray.HIDDEN_FIELDS
    assert "recipient_mass_overlap" in xray.HIDDEN_FIELDS


def test_the_control_logs_its_cap_and_the_mass_landing_on_flagged_positives(
    tmp_path: Path,
) -> None:
    history = _masses(tmp_path, "matched_random")
    assert all(0.0 <= h["capped_share"] <= 1.0 for h in history)
    # In this tiny run, some epochs have no recipients, and log no overlap.
    logged = [h for h in history if h["recipient_overlap"] is not None]
    assert logged
    assert all(0.0 <= h["recipient_mass_overlap"] <= 1.0 for h in logged)
    # Flagged positives hold more gradient, so the mass overlap is at least the count overlap.
    assert any(h["recipient_mass_overlap"] >= h["recipient_overlap"] for h in logged)


def test_arms_without_a_boost_log_no_extra_gradient(tmp_path: Path) -> None:
    history = _train(tmp_path, _run("ce", seed=6))["history"]
    assert all(h["extra_mass"] is None for h in history)


def test_the_control_delivers_the_mass_it_was_asked_for() -> None:
    base = AUCReshapingLoss([0.5], boost=[0.2], min_negatives=4)
    control = xray.MatchedBoost(base, "random", 1)
    control.train()
    y = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    z = torch.tensor([-1.0, 0.0, 2.0, 3.0, -2.0, -1.0, 0.0, 1.0])
    control.tracker.update(z, y)
    control(z.clone(), y)
    assert not control.last_capped
    assert control.last_delivered == pytest.approx(control.last_extra)
    assert control.last_extra > 0.0


def test_unknown_or_incomplete_chosen_settings_are_refused() -> None:
    no_ar = {"weighted_bce": {"factor": 3.0}}
    with pytest.raises(ValueError, match="aucreshape"):
        xray.confirmatory_jobs("pc", 1e-4, no_ar, [2000], "s")
    empty_ar = {"aucreshape": {}, "weighted_bce": {"factor": 3.0}}
    with pytest.raises(ValueError, match="combine"):
        xray.confirmatory_jobs("pc", 1e-4, empty_ar, [2000], "s")
    no_floor = {"aucreshape": {"combine": "band"}, "weighted_bce": {"factor": 3.0}}
    with pytest.raises(ValueError, match="min_ratio"):
        xray.confirmatory_jobs("pc", 1e-4, no_floor, [2000], "s")
    no_factor = {"aucreshape": CHOSEN["aucreshape"], "weighted_bce": {}}
    with pytest.raises(ValueError, match="factor"):
        xray.confirmatory_jobs("pc", 1e-4, no_factor, [2000], "s")


def test_the_contrast_setting_cannot_duplicate_the_chosen_one() -> None:
    chosen = {"aucreshape": dict(xray.CONTRAST_SETTING), "weighted_bce": {"factor": 3.0}}
    with pytest.raises(ValueError, match="duplicate"):
        xray.confirmatory_jobs("pc", 1e-4, chosen, [2000], "s")


def test_the_frozen_choice_is_what_confirmatory_jobs_read(tmp_path: Path) -> None:
    rows = _tuned_rows({1.25: 0.30, 3.0: 0.33})
    gentle = dict(CHOSEN["aucreshape"])
    rows += [_tuned_row("aucreshape", seed, gentle, 0.31) for seed in xray.TUNING_SEEDS]
    path = tmp_path / "chosen.json"
    xray.write_chosen(rows, path)
    chosen = json.loads(path.read_text())
    assert chosen == {"aucreshape": CHOSEN["aucreshape"], "weighted_bce": {"factor": 3.0}}
    assert xray.confirmatory_jobs("pc", 1e-4, chosen, [2000], "s")


def test_the_frozen_choice_needs_both_arms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aucreshape"):
        xray.write_chosen(_tuned_rows({1.25: 0.3}), tmp_path / "chosen.json")


def test_confirmatory_kernels_write_the_expected_runs_without_arms(tmp_path: Path) -> None:
    jobs = xray.confirmatory_jobs("pc", 1e-4, CHOSEN, range(2000, 2002), "salt")
    xray.confirmatory_kernels(tmp_path, "someone", jobs, ("a/code",), ("a/prep",))
    expected = json.loads((tmp_path / "expected_runs.json").read_text())
    assert sorted(expected) == sorted(job.key for job in jobs)
    assert "aucreshape" not in json.dumps(expected)


def test_coded_runs_save_the_no_finding_labels(tmp_path: Path) -> None:
    row = _coded(tmp_path)
    with np.load(tmp_path / "out" / f"{row['key']}.npz") as saved:
        assert saved["no_finding_test"].shape == saved["y_test"].shape


def test_validation_only_data_does_not_load_no_finding_test_labels(tmp_path: Path) -> None:
    root = tmp_path / "nih"
    _fake_root(root)
    assert xray.load_data(root, "Effusion", include_test=False).no_finding_test is None


def test_a_run_records_a_fingerprint_of_its_data(tmp_path: Path) -> None:
    root = tmp_path / "nih"
    _fake_root(root)
    first = xray.data_fingerprint(root, include_test=False)
    assert set(first) == {"labels_trainval.npz", "images_trainval.npy"}
    with_test = xray.data_fingerprint(root, include_test=True)
    assert set(with_test) == {
        "labels_trainval.npz",
        "images_trainval.npy",
        "labels_test.npz",
        "images_test.npy",
    }
    assert with_test["labels_trainval.npz"] == first["labels_trainval.npz"]
    (root / "labels_trainval.npz").write_bytes(b"changed")
    assert xray.data_fingerprint(root, False)["labels_trainval.npz"] != first["labels_trainval.npz"]


def test_validation_only_runs_do_not_open_test_files_for_the_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "nih"
    _fake_root(root, with_test=False)
    assert "labels_test.npz" not in xray.data_fingerprint(root, include_test=False)
