"""The synthetic laboratory redone with fine-tuning, in the paper's regime too (exploratory)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from benchmarks.review import finetune_lab as lab

FAST = lab.Settings(
    seeds=(1900, 1901),
    tune_seeds=(1910,),
    pretrain_epochs=4,
    finetune_epochs=3,
    n_boot=10,
    scale=0.02,
    ar_grid=lab.AR_GRID[:2],
    weight_grid=lab.WEIGHT_GRID[:2],
    bce_grid=lab.BCE_GRID[:2],
)


def test_design_uses_fresh_seeds_and_equal_grids() -> None:
    settings = lab.Settings()
    assert len(settings.seeds) == 10
    assert set(settings.seeds).isdisjoint(settings.tune_seeds)
    assert min(settings.seeds) >= 1900
    assert max(*settings.seeds, *settings.tune_seeds) < 2000
    assert len(lab.AR_GRID) == len(lab.BCE_GRID) == len(lab.WEIGHT_GRID) == 6


def test_the_grid_holds_the_papers_setting_and_reaches_gentle_boosts() -> None:
    ratios = {g["min_ratio"] for g in lab.AR_GRID}
    assert 0.1 in ratios
    assert 0.75 in ratios
    assert {g["combine"] for g in lab.AR_GRID} == {"band", "sum"}
    assert min(lab.BCE_GRID) < 1.5


def test_the_plan_covers_every_level_in_both_regimes() -> None:
    plan = lab.plan()
    assert len(plan) == 60
    assert {job[2] for job in plan} == {"rare", "paper"}
    assert len({(job[0], lab.level_key(job[1]), job[2]) for job in plan}) == 60


def test_shards_partition_the_plan() -> None:
    plan = lab.plan()
    parts = [plan[i::4] for i in range(4)]
    assert sum(len(p) for p in parts) == 60


def _view(regime: str) -> lab.View:
    from benchmarks.review import e1

    return lab.view(e1.EXPERIMENTS["S-3"].task(0.0, 1900, 0.02), regime)


def test_paper_regime_boosts_the_majority_with_class_weights() -> None:
    rare, paper = _view("rare"), _view("paper")
    assert float(rare.y_train.mean()) < 0.5
    assert float(paper.y_train.mean()) > 0.5
    assert torch.equal(paper.y_test, 1 - rare.y_test)
    assert float(rare.class_weight) == 1.0
    assert float(paper.class_weight) < 1.0
    assert float(paper.class_weight) == pytest.approx(
        float((1 - paper.y_train).sum() / paper.y_train.sum())
    )


def test_unknown_regime_is_refused() -> None:
    from benchmarks.review import e1

    with pytest.raises(ValueError, match="regime"):
        lab.view(e1.EXPERIMENTS["S-3"].task(0.0, 1900, 0.02), "other")


def test_batches_always_give_enough_steps() -> None:
    assert lab.finetune_batch(200) == 64
    assert lab.finetune_batch(2000) == 100
    assert lab.finetune_batch(20000) == 1000
    assert lab.finetune_batch(100000) == 1024


def _start(regime: str = "paper") -> tuple[lab.View, object, dict]:
    v = _view(regime)
    spec = lab.base_spec("S-3", 0.0, 1900)
    state, info = lab.pretrain(v, spec, 1900, FAST)
    return v, spec, {"state": state, "info": info}


def test_pretraining_keeps_the_best_validation_checkpoint() -> None:
    _, _, out = _start()
    assert out["info"]["best_epoch"] in range(1, FAST.pretrain_epochs + 1)
    assert out["info"]["val_loss"] > 0


def test_every_arm_fine_tunes_from_the_same_checkpoint() -> None:
    v, spec, out = _start()
    params = lab.AR_GRID[0]
    # With no extra steps, every arm is still the pretrained model.
    zero = lab.Settings(**{**FAST.__dict__, "finetune_epochs": 0})
    scores = [
        lab.finetune(v, spec, out["state"], arm, p, 1900, zero)["val_at_start"]
        for arm, p in (("ce", {}), ("aucreshape", params), ("matched_random", params))
    ]
    assert scores[0].equal(scores[1])
    assert scores[0].equal(scores[2])


def test_arms_share_batch_order_a_zero_boost_is_the_baseline() -> None:
    v, spec, out = _start()
    ce = lab.finetune(v, spec, out["state"], "ce", {}, 1900, FAST)
    zero = lab.finetune(
        v, spec, out["state"], "aucreshape", {"boost": [0.0, 0.0, 0.0, 0.0]}, 1900, FAST
    )
    assert torch.allclose(ce["test"], zero["test"], rtol=1e-3, atol=1e-3)


def test_selection_never_reads_test_data() -> None:
    v, spec, out = _start()
    params = lab.AR_GRID[0]
    first = lab.finetune(v, spec, out["state"], "aucreshape", params, 1900, FAST)
    noise = lab.View(
        v.x_train, v.y_train, v.x_val, v.y_val, torch.randn_like(v.x_test), 1 - v.y_test,
        v.class_weight,
    )  # fmt: skip
    second = lab.finetune(noise, spec, out["state"], "aucreshape", params, 1900, FAST)
    assert first["epoch"] == second["epoch"]
    assert first["val_sens"] == second["val_sens"]


def test_a_finetuned_arm_reports_boosting() -> None:
    v, spec, out = _start()
    row = lab.finetune(v, spec, out["state"], "matched_random", lab.AR_GRID[0], 1900, FAST)
    assert row["boosted_fraction"] is not None
    assert row["recipient_overlap"] is not None
    assert row["test"].shape == v.y_test.shape


def test_tuning_chooses_on_validation_from_the_arms_own_grid() -> None:
    chosen = lab.tune("S-3", 0.0, "paper", FAST)
    assert chosen["aucreshape"] in list(FAST.ar_grid)
    assert chosen["weighted_bce"]["factor"] in FAST.bce_grid
    assert chosen["ar_weight"] in list(FAST.weight_grid)


@pytest.mark.parametrize("regime", ["rare", "paper"])
def test_one_level_gives_every_contrast_with_the_permitted_wording(regime: str) -> None:
    row = lab.run_level("S-3", 0.0, regime, FAST)
    assert row["status"] == "ok"
    assert row["regime"] == regime
    assert set(row["arms"]) == set(lab.ARMS)
    assert all(len(values) == 2 for values in row["arms"].values())
    named = {c["minus"] for c in row["contrasts"]}
    assert named == {
        "ce", "weighted_bce", "matched_random", "same_boost_random", "matched_above",
    }  # fmt: skip
    for contrast in row["contrasts"]:
        assert contrast["lo"] <= contrast["hi"]
        assert isinstance(contrast["claim"], str)
    assert "strength" in row["diagnostics"]
    json.dumps(row)


def test_a_failed_level_is_recorded_not_raised() -> None:
    row = lab.safe_level(("S-3", "no such level", "rare"), FAST)
    assert row["status"].startswith("failed")


def test_kernel_is_private_cpu_and_offline(tmp_path: Path) -> None:
    folder = lab.make_kernel(tmp_path, "someone", 1, 4, ("someone/code",))
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/aucreshape-s13c-lab-1"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is False
    assert meta["dataset_sources"] == ["someone/code"]
    compile((folder / meta["code_file"]).read_text(), "kernel", "exec")


def test_the_standard_grid_stays_as_it_ran() -> None:
    assert lab.grid_settings("standard") == lab.Settings()


def test_the_extended_grids_keep_every_old_setting_and_pass_both_edges() -> None:
    wide = lab.grid_settings("extended")
    assert wide.name == "extended"
    assert len(wide.ar_grid) == len(wide.weight_grid) == len(wide.bce_grid) == 9
    assert all(setting in wide.ar_grid for setting in lab.AR_GRID)
    assert all(setting in wide.weight_grid for setting in lab.WEIGHT_GRID)
    assert set(lab.BCE_GRID) < set(wide.bce_grid)
    assert min(wide.bce_grid) < min(lab.BCE_GRID)
    assert max(wide.bce_grid) > max(lab.BCE_GRID)
    ratios = [g["min_ratio"] for g in wide.ar_grid]
    assert max(ratios) > max(g["min_ratio"] for g in lab.AR_GRID)
    assert min(ratios) < min(g["min_ratio"] for g in lab.AR_GRID)
    boosts = [g["boost"][0] for g in wide.weight_grid]
    assert min(boosts) < min(g["boost"][0] for g in lab.WEIGHT_GRID)
    assert max(boosts) > max(g["boost"][0] for g in lab.WEIGHT_GRID)


def test_an_unknown_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="grid"):
        lab.grid_settings("huge")


@pytest.mark.parametrize(
    ("arm", "params"),
    [
        ("aucreshape", lab.EXTENDED_AR_GRID[0]),
        ("aucreshape", lab.EXTENDED_AR_GRID[-1]),
        ("matched_random", lab.EXTENDED_AR_GRID[-1]),
        ("matched_above", lab.EXTENDED_AR_GRID[-1]),
        ("ar_weight", lab.EXTENDED_WEIGHT_GRID[0]),
        ("ar_weight", lab.EXTENDED_WEIGHT_GRID[-1]),
        ("weighted_bce", {"factor": lab.EXTENDED_BCE_GRID[-1]}),
    ],
)
def test_the_extreme_settings_stay_finite(arm: str, params: dict[str, object]) -> None:
    v, spec, state, _ = lab._prepare("S-3", 0.02, "rare", 1900, FAST)
    out = lab.finetune(v, spec, state, arm, params, 1900, FAST)
    assert torch.isfinite(out["test"]).all()


def test_a_row_records_which_grid_ran() -> None:
    settings = dataclasses.replace(FAST, name="extended")
    row = lab.run_level("S-3", 0.02, "rare", settings)
    assert row["grid"] == "extended"


def test_an_extended_kernel_has_its_own_name_and_passes_the_grid(tmp_path: Path) -> None:
    folder = lab.make_kernel(tmp_path, "someone", 2, 5, ("someone/code",), grid="extended")
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/aucreshape-s13c-lab-extended-2"
    code = (folder / meta["code_file"]).read_text()
    assert '"--grid", "extended"' in code
    compile(code, "kernel", "exec")
