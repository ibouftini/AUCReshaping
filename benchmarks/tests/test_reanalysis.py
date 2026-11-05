"""Direct contrasts and noise floor from saved results (exploratory)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.review import reanalysis as ra

WEIGHT = {"boost": 16.0, "mode": "weight"}
PROB = {"boost": 0.4}
CONTROLS = ("random_boost",)


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (1.2, 3.0, "gain of at least about 1 point"),
        (0.2, 1.5, "gain, size uncertain"),
        (-0.8, 0.9, "no difference larger than 1 point"),
        (0.2, 0.9, "no difference larger than 1 point"),
        (-0.5, 1.5, "inconclusive"),
        (-1.5, 0.5, "inconclusive"),
        (-1.5, 1.5, "inconclusive"),
        (-1.5, -0.2, "loss, size uncertain"),
        (-3.0, -1.2, "loss of at least about 1 point"),
    ],
)
def test_claim_label_follows_the_fixed_wording(lo: float, hi: float, expected: str) -> None:
    assert ra.claim_label(lo, hi) == expected


def test_claim_label_treats_the_limits_as_inside_the_region() -> None:
    assert ra.claim_label(-1.0, 1.0) == "no difference larger than 1 point"


def test_claim_label_rejects_a_reversed_interval() -> None:
    with pytest.raises(ValueError, match="lo"):
        ra.claim_label(1.0, -1.0)


def _chosen(folder: Path, final: dict, original: dict | None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    extra = {"weighted_bce": {"pos_weight": 8.0}, "focal": {"gamma": 2.0}}
    (folder / "chosen.json").write_text(json.dumps({"aucreshape": final, **extra}))
    if original is not None:
        (folder / "chosen_original.json").write_text(json.dumps({"aucreshape": original, **extra}))


def test_settings_split_prob_from_weight(tmp_path: Path) -> None:
    _chosen(tmp_path / "S-3", WEIGHT, PROB)
    settings = ra.read_settings(tmp_path / "S-3")
    assert sorted(settings) == ["prob", "weight"]
    assert settings["weight"]["aucreshape"] == WEIGHT
    assert settings["prob"]["aucreshape"] == PROB


def test_settings_with_one_prob_choice(tmp_path: Path) -> None:
    _chosen(tmp_path / "S-1", {"boost": 0.1}, None)
    assert sorted(ra.read_settings(tmp_path / "S-1")) == ["prob"]


def test_settings_refuse_two_choices_of_one_mode(tmp_path: Path) -> None:
    _chosen(tmp_path / "S-9", PROB, {"boost": 0.2})
    with pytest.raises(ValueError, match="prob"):
        ra.read_settings(tmp_path / "S-9")


def test_labels_survive_packing() -> None:
    y = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1], dtype=np.float32)
    assert np.array_equal(ra.unpack_labels(ra.pack_labels(y), len(y)), y > 0.5)


@pytest.mark.parametrize("wrong", [16, 17, 8])
def test_unpacking_refuses_a_wrong_length(wrong: int) -> None:
    # np.unpackbits pads with zeros when asked for too many bits, which would silently corrupt
    # labels; 10 labels fill 2 bytes, so 16 bits passes the byte count but not the stored size.
    packed = ra.pack_labels(np.ones(10, dtype=np.float32))
    if wrong == 16:
        assert len(ra.unpack_labels(packed, 10)) == 10
    else:
        with pytest.raises(ValueError, match="labels"):
            ra.unpack_labels(packed, wrong)


def test_validation_threshold_reads_sensitivity_and_realised_specificity() -> None:
    y_val = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    val = np.array([0.1, 0.2, 0.3, 0.4, 0.9], dtype=np.float32)
    y_test = np.array([0, 0, 0, 0, 1, 1], dtype=np.float32)
    test = np.array([0.1, 0.2, 0.3, 0.5, 0.45, 0.2], dtype=np.float32)
    sens, spec = ra.validation_threshold(val, y_val, test, y_test, specificity=0.75)
    # 3 of 4 validation negatives lie at or below the cut, which is therefore 0.3.
    assert sens == pytest.approx(0.5)
    assert spec == pytest.approx(0.75)


def test_noise_floor_counts_every_pair() -> None:
    floor = ra.noise_floor([10.0, 11.0, 13.0])
    assert floor["pairs"] == 3
    assert floor["median_abs"] == pytest.approx(2.0)
    assert floor["max_abs"] == pytest.approx(3.0)


def test_noise_floor_of_two_values_is_their_difference() -> None:
    floor = ra.noise_floor([26.0, 25.3])
    assert floor["pairs"] == 1
    assert floor["max_abs"] == pytest.approx(0.7)


def _run(name: str, tag: str, arm: str, params: dict, seed: int, key: str) -> dict:
    history = [
        {"epoch": 10, "val_loss": 0.2, "val_sens": 0.3},
        {"epoch": 20, "val_loss": 0.1, "val_sens": 0.2},
    ]
    return {
        "key": key,
        "status": "ok",
        "spec": {
            "task": f"{name}/{tag}",
            "arm": arm,
            "params": params,
            "seed": seed,
            "checkpoints": [10, 20],
        },
        "history": history,
    }


def _results(root: Path, gain: float = 0.5) -> dict:
    """A small results tree: AUCReshaping's scores separate the classes better than the control."""
    name, tag = "S-3", "0.0"
    _chosen(root / name, WEIGHT, PROB)
    rng = np.random.default_rng(0)
    labels = {}
    for seed in range(4):
        y = (rng.random(2000) < 0.3).astype(np.float32)
        y_val = (rng.random(500) < 0.3).astype(np.float32)
        labels[f"{name}|{tag}|{seed}|test"] = ra.pack_labels(y)
        labels[f"{name}|{tag}|{seed}|val"] = ra.pack_labels(y_val)
        folder = root / name / tag / f"seed{seed}"
        (folder / "scores").mkdir(parents=True)
        lines = []
        arms = [
            ("aucreshape", WEIGHT, gain),
            ("random_boost", WEIGHT, 0.0),
            ("aucreshape", PROB, 0.0),
            ("random_boost", PROB, 0.0),
            ("weighted_bce", {"pos_weight": 8.0}, 0.0),
            ("ce", {}, 0.0),
        ]
        for i, (arm, params, extra) in enumerate(arms):
            key = f"{arm}{i}"
            noise = rng.normal(size=2000)
            scores = {
                "test_10": noise + (1.0 + extra) * y,
                "test_20": noise,
                "val_10": rng.normal(size=500) + y_val,
                "val_20": rng.normal(size=500),
            }
            np.savez(
                folder / "scores" / f"{key}.npz",
                **{k: v.astype(np.float16) for k, v in scores.items()},
            )
            lines.append(json.dumps(_run(name, tag, arm, params, seed, key)))
        (folder / "runs.jsonl").write_text("\n".join(lines) + "\n")
    # Sizes are per array: the scenarios do not share one validation size.
    meta = {key: 2000 if key.endswith("test") else 500 for key in labels}
    return {"labels": labels, "meta": meta}


def test_index_finds_each_setting_with_its_own_controls(tmp_path: Path) -> None:
    _results(tmp_path)
    index = ra.build_index(tmp_path, CONTROLS)
    runs = index["S-3"]["weight"]["0.0"]
    assert sorted(runs) == ["aucreshape", "ce", "random_boost", "weighted_bce"]
    assert sorted(runs["aucreshape"]) == ["0", "1", "2", "3"]
    assert runs["aucreshape"]["0"]["key"] == "aucreshape0"
    assert index["S-3"]["prob"]["0.0"]["aucreshape"]["0"]["key"] == "aucreshape2"


def test_index_records_the_selected_epochs(tmp_path: Path) -> None:
    _results(tmp_path)
    run = ra.build_index(tmp_path, CONTROLS)["S-3"]["prob"]["0.0"]["ce"]["0"]
    assert (run["val"], run["loss"], run["last"]) == (10, 20, 20)


def test_contrasts_compare_aucreshape_with_every_control_and_weighted_bce(tmp_path: Path) -> None:
    _results(tmp_path)
    plan = ra.contrasts(ra.build_index(tmp_path, CONTROLS))
    assert {(c.setting, c.arm) for c in plan} == {
        ("weight", "random_boost"),
        ("weight", "weighted_bce"),
        ("prob", "random_boost"),
        ("prob", "weighted_bce"),
    }


def test_estimate_is_aucreshape_minus_the_control_in_points(tmp_path: Path) -> None:
    built = _results(tmp_path, gain=1.0)
    index = ra.build_index(tmp_path, CONTROLS)
    row = ra.estimate(
        ra.Contrast("S-3", "0.0", "weight", "random_boost"),
        index,
        tmp_path,
        built["labels"],
        built["meta"],
        n_boot=20,
    )
    assert row["delta"] > 5
    assert row["ar_sens"] > row["control_sens"]
    assert row["lo"] <= row["delta"] <= row["hi"]
    assert len(row["per_seed"]) == 4
    assert row["claim"] == "gain of at least about 1 point"
    assert row["paired_sd"] == pytest.approx(np.std(row["per_seed"], ddof=1))


def test_estimate_of_identical_arms_is_zero(tmp_path: Path) -> None:
    built = _results(tmp_path)
    index = ra.build_index(tmp_path, CONTROLS)
    runs = index["S-3"]["prob"]["0.0"]
    runs["random_boost"] = runs["aucreshape"]
    row = ra.estimate(
        ra.Contrast("S-3", "0.0", "prob", "random_boost"),
        index,
        tmp_path,
        built["labels"],
        built["meta"],
        n_boot=20,
    )
    assert row["delta"] == 0.0
    assert row["claim"] == "no difference larger than 1 point"


def test_shards_partition_the_contrasts(tmp_path: Path) -> None:
    _results(tmp_path)
    plan = ra.contrasts(ra.build_index(tmp_path, CONTROLS))
    parts = [ra.shard(plan, k, 3) for k in range(3)]
    assert sum(len(p) for p in parts) == len(plan)
    assert len({c for p in parts for c in p}) == len(plan)


def test_kernel_is_private_cpu_and_offline(tmp_path: Path) -> None:
    folder = ra.make_kernel(tmp_path, "someone", 1, 5, ("someone/bundle", "someone/eval"))
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/aucreshape-s13a-contrasts-1"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is False
    assert meta["dataset_sources"] == ["someone/bundle", "someone/eval"]
    code = (folder / meta["code_file"]).read_text()
    compile(code, "kernel", "exec")
    assert "run_kernel(1, 5)" in code


@pytest.mark.parametrize("excluded", ["tune", "interventions"])
def test_index_ignores_tuning_and_intervention_runs(tmp_path: Path, excluded: str) -> None:
    _results(tmp_path)
    folder = tmp_path / "S-3" / excluded / "0.0" / "seed0"
    folder.mkdir(parents=True)
    stray = _run("S-3", "0.0", "aucreshape", PROB, 0, "stray")
    (folder / "runs.jsonl").write_text(json.dumps(stray) + "\n")
    index = ra.build_index(tmp_path, CONTROLS)
    assert index["S-3"]["prob"]["0.0"]["aucreshape"]["0"]["key"] == "aucreshape2"
