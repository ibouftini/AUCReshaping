"""The confirmatory analysis of the chest X-ray study, tested on made-up runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.review import reanalysis
from benchmarks.review import xray_analysis as xa

SEEDS = (2000, 2001, 2002, 2003, 2004)
CODES = {
    "ce": "arm-1",
    "aucreshape": "arm-2",
    "weighted_bce": "arm-4",
    "aucreshape_6x": "arm-5",
    "matched_random_6x": "arm-3",
    "matched_above_6x": "arm-6",
}
PATIENTS = 400


def _labels(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Disease is 1, as saved; three images per patient, about 30% disease."""
    rng = np.random.default_rng(seed)
    patients = np.repeat(np.arange(PATIENTS), 3)
    disease = (rng.random(PATIENTS) < 0.3).astype(np.uint8)[patients]
    return disease, patients


def _write_run(
    root: Path,
    code: str,
    seed: int,
    gain: float,
    status: str = "ok",
    best_epoch: int = 2,
) -> None:
    """Write one coded run. A positive gain moves normals away from disease, the paper's aim."""
    folder = root / "worker-0"
    folder.mkdir(parents=True, exist_ok=True)
    key = f"pc_{code}_{seed}"
    row: dict[str, Any] = {"key": key, "status": status, "config": {"code": code, "seed": seed}}
    if status == "ok":
        row["history"] = [{"epoch": e, "val_sens": 0.3, "val_loss": 0.1} for e in (1, 2, 3)]
        row["selected"] = {"val_sens": best_epoch, "val_loss": 1, "last": 3}
        y, patients = _labels()
        y_val, _ = _labels(1)
        rng = np.random.default_rng(seed)
        arrays: dict[str, np.ndarray] = {"y_test": y, "patients_test": patients, "y_val": y_val}
        for epoch in (1, 2, 3):
            # Scores are disease logits. Epoch 2 is good, others are noise.
            sep = 2.0 if epoch == 2 else 0.0
            common = rng.normal(0, 1.0, len(y))
            test = common + sep * y - (gain * (1 - y) if epoch == 2 else 0.0)
            val = rng.normal(0, 1.0, len(y_val)) + sep * y_val
            arrays[f"test_{epoch}"] = test.astype(np.float32)
            arrays[f"val_{epoch}"] = val.astype(np.float32)
        np.savez(folder / f"{key}.npz", **arrays)
    with (folder / "runs.jsonl").open("a") as log:
        log.write(json.dumps(row) + "\n")


def _study(root: Path, gains: dict[str, float]) -> Path:
    for arm, code in CODES.items():
        for seed in SEEDS:
            _write_run(root, code, seed, gains.get(arm, 0.0))
    return root


def _analyse(root: Path, **kw: Any) -> dict[str, Any]:
    return xa.analyse(xa.load_study(root), CODES, n_boot=kw.pop("n_boot", 50), **kw)


def test_scores_are_read_in_the_papers_direction(tmp_path: Path) -> None:
    # Normals scored far from disease read as high sensitivity at high specificity.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    scores = np.array([3.0, 3.1, 3.2, 3.3, -3.0, -3.1, -3.2, -3.3], dtype=np.float32)
    assert xa.sens_points(scores, y) == pytest.approx(100.0)
    assert xa.sens_points(-scores, y) == pytest.approx(0.0)


def test_a_run_is_read_at_its_own_selected_epoch(tmp_path: Path) -> None:
    good = tmp_path / "good"
    _write_run(good, "arm-1", 2000, 0.0, best_epoch=2)
    bad = tmp_path / "bad"
    _write_run(bad, "arm-1", 2000, 0.0, best_epoch=3)
    high = xa.read_run(xa.load_study(good)[0], "val_sens")
    low = xa.read_run(xa.load_study(bad)[0], "val_sens")
    assert high["epoch"] == 2
    assert low["epoch"] == 3
    assert high["sens_roc"] > low["sens_roc"] + 10.0


def test_a_run_reports_both_readouts(tmp_path: Path) -> None:
    _write_run(tmp_path, "arm-1", 2000, 0.0)
    out = xa.read_run(xa.load_study(tmp_path)[0], "val_sens")
    for field in ("sens_roc", "sens_at_val_threshold", "realised_specificity", "auc", "pauc"):
        assert np.isfinite(out[field])
    assert 0.5 < out["auc"] <= 1.0
    assert 0.0 <= out["realised_specificity"] <= 100.0


def test_every_primary_contrast_is_reported_for_each_rule(tmp_path: Path) -> None:
    report = _analyse(_study(tmp_path, {"aucreshape": 0.5}))
    for rule in ("val_sens", "val_loss"):
        assert [c["name"] for c in report["rules"][rule]["contrasts"]] == ["P1", "P2", "P3"]
    assert report["primary_rule"] == "val_sens"
    first = report["rules"]["val_sens"]["contrasts"][0]
    assert first["ours"] == "aucreshape_6x"
    assert first["minus"] == "matched_random_6x"
    assert len(first["per_seed"]) == len(SEEDS)
    assert first["claim"] in xa.WORDINGS


def test_a_clear_gain_reads_as_a_gain_and_identical_arms_as_no_difference(tmp_path: Path) -> None:
    root = tmp_path / "study"
    for arm, code in CODES.items():
        for seed in SEEDS:
            _write_run(root, code, seed, 2.0 if arm == "aucreshape" else 0.0)
    report = _analyse(root, n_boot=100)
    p3 = report["rules"]["val_sens"]["contrasts"][2]
    assert p3["delta"] > 1.0
    assert p3["lo"] > 0.0
    assert p3["seeds_with_gain"] == len(SEEDS)


def test_the_interval_is_the_wider_of_bootstrap_and_paired_t(tmp_path: Path) -> None:
    report = _analyse(_study(tmp_path, {"aucreshape": 0.5}))
    contrast = report["rules"]["val_sens"]["contrasts"][0]
    assert contrast["lo"] == min(contrast["lo_bootstrap"], contrast["lo_t"])
    assert contrast["hi"] == max(contrast["hi_bootstrap"], contrast["hi_t"])


def test_the_interval_is_split_into_seeds_and_patients(tmp_path: Path) -> None:
    contrast = _analyse(_study(tmp_path, {"aucreshape": 0.5}))["rules"]["val_sens"]["contrasts"][0]
    assert contrast["width_seeds"] >= 0.0
    assert contrast["width_patients"] >= 0.0


def test_holm_applies_across_the_primary_family_only() -> None:
    assert xa.family_claims([(0.2, 3.0, 0.01), (0.2, 3.0, 0.04), (-0.5, 0.5, 0.9)]) == [
        "gain, size uncertain",
        "inconclusive",
        "no difference larger than 1 point",
    ]


def test_holm_cannot_rescue_or_remove_an_equivalence_claim() -> None:
    assert xa.family_claims([(-0.5, 0.5, 1.0)]) == ["no difference larger than 1 point"]
    assert xa.family_claims([(-0.2, 3.0, 0.2)]) == ["inconclusive"]


def test_a_loss_keeps_its_wording_when_it_survives_holm() -> None:
    assert xa.family_claims([(-3.0, -1.5, 0.001)]) == ["loss of at least about 1 point"]


def test_a_failed_run_is_shown_and_its_seed_leaves_only_the_contrasts_it_touches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study"
    for arm, code in CODES.items():
        for seed in SEEDS:
            failed = arm == "weighted_bce" and seed == 2003
            _write_run(root, code, seed, 0.0, status="failed" if failed else "ok")
    report = _analyse(root)
    assert report["failed"] == [{"arm": "weighted_bce", "seed": 2003}]
    contrasts = {c["name"]: c for c in report["rules"]["val_sens"]["contrasts"]}
    assert contrasts["P2"]["seeds"] == [s for s in SEEDS if s != 2003]
    assert contrasts["P1"]["seeds"] == list(SEEDS)


def test_every_seed_is_shown_for_every_arm(tmp_path: Path) -> None:
    report = _analyse(_study(tmp_path, {}))
    arms = report["rules"]["val_sens"]["arms"]
    assert set(arms) == set(CODES)
    assert [row["seed"] for row in arms["ce"]] == list(SEEDS)


def test_an_unknown_code_is_refused(tmp_path: Path) -> None:
    _write_run(tmp_path, "arm-9", 2000, 0.0)
    with pytest.raises(ValueError, match="arm-9"):
        xa.analyse(xa.load_study(tmp_path), CODES, n_boot=2)


def test_runs_that_disagree_on_the_test_set_are_refused(tmp_path: Path) -> None:
    _write_run(tmp_path, "arm-1", 2000, 0.0)
    _write_run(tmp_path, "arm-2", 2000, 0.0)
    study = xa.load_study(tmp_path)
    study[1]["arrays"]["y_test"] = 1 - study[1]["arrays"]["y_test"]
    with pytest.raises(ValueError, match="test set"):
        xa.analyse(study, {"ce": "arm-1", "aucreshape": "arm-2"}, n_boot=2)


def test_integrity_reads_no_results(tmp_path: Path) -> None:
    root = _study(tmp_path / "study", {})
    report = xa.integrity(xa.load_study(root))
    assert report["runs"] == len(CODES) * len(SEEDS)
    assert report["codes"] == sorted(CODES.values())
    assert report["seeds"] == list(SEEDS)
    assert report["not_ok"] == []
    assert report["not_finite"] == []
    # It says nothing that would separate the arms.
    assert "sens" not in json.dumps(report)


def test_the_report_is_ascii_and_states_the_scope(tmp_path: Path) -> None:
    text = xa.markdown(_analyse(_study(tmp_path, {"aucreshape": 0.5})))
    text.encode("ascii")
    assert "P1" in text
    assert "98%" in text


def test_the_report_gives_the_noise_floor_and_the_cases_behind_the_threshold(
    tmp_path: Path,
) -> None:
    report = _analyse(_study(tmp_path, {}))
    y, _ = _labels()
    assert report["cases_behind_threshold"] == pytest.approx(int(y.sum()) * 0.02)
    floor = report["rules"]["val_sens"]["noise_floor"]
    assert floor["pairs"] == 10
    assert floor["median_abs"] >= 0.0


def test_the_simulation_has_a_known_answer_and_returns_coverage() -> None:
    out = xa.simulate_coverage(seeds=4, sims=3, n_boot=20, patients=60, seed=1)
    assert out["seeds"] == 4
    assert out["sims"] == 3
    assert 0.0 < out["true_effect"] < 100.0
    for field in ("bootstrap", "t", "conservative"):
        assert 0.0 <= out[field] <= 1.0
    assert out["conservative"] >= max(out["bootstrap"], out["t"])


def test_the_validation_threshold_comes_from_validation_scores() -> None:
    rng = np.random.default_rng(3)
    y, patients = _labels()
    test = (rng.normal(0, 1.0, len(y)) + 2.0 * y).astype(np.float32)
    # The same model, but validation scores sit far lower, so its threshold is far stricter on test.
    val = (rng.normal(0, 1.0, len(y)) + 2.0 * y - 3.0).astype(np.float32)
    row = {
        "config": {"seed": 2000},
        "selected": {"val_sens": 1},
        "arrays": {
            "test_1": test,
            "val_1": val,
            "y_test": y,
            "y_val": y,
            "patients_test": patients,
        },
    }
    out = xa.read_run(row, "val_sens")
    assert out["realised_specificity"] > 99.9
    assert out["sens_at_val_threshold"] < out["sens_roc"]


def _sealed(
    root: Path, code: str, seed: int, arm: str, share: float, overlap: float | None
) -> None:
    folder = root / "worker-0" / "sealed"
    folder.mkdir(parents=True, exist_ok=True)
    history = [
        {"epoch": e, "boosted_fraction": share * e, "recipient_overlap": overlap} for e in (1, 2)
    ]
    content = {"arm": arm, "params": {}, "history": history}
    (folder / f"pc_{code}_{seed}.json").write_text(json.dumps(content))


def test_the_share_boosted_and_the_overlap_are_read_from_the_sealed_files(tmp_path: Path) -> None:
    _sealed(tmp_path, "arm-2", 2000, "aucreshape", 0.2, None)
    _sealed(tmp_path, "arm-2", 2001, "aucreshape", 0.4, None)
    _sealed(tmp_path, "arm-3", 2000, "matched_random", 0.1, 0.6)
    out = xa.boost_summary(tmp_path)
    assert out["aucreshape"]["boosted_by_epoch"] == pytest.approx([0.3, 0.6])
    assert out["aucreshape"]["overlap_by_epoch"] is None
    assert out["matched_random"]["overlap_by_epoch"] == pytest.approx([0.6, 0.6])


def test_a_failed_runs_sealed_file_is_left_out_of_the_summary(tmp_path: Path) -> None:
    folder = tmp_path / "worker-0" / "sealed"
    folder.mkdir(parents=True)
    (folder / "x.json").write_text(json.dumps({"arm": "ce", "params": {}, "error": "boom"}))
    assert xa.boost_summary(tmp_path) == {}


def test_the_contrast_setting_has_its_own_secondary_contrasts(tmp_path: Path) -> None:
    root = tmp_path / "study"
    _study(root, {"aucreshape_6x": 1.5})
    rule = xa.analyse(xa.load_study(root), CODES, n_boot=50)["rules"]["val_sens"]
    assert [c["name"] for c in rule["contrasts"]] == ["P1", "P2", "P3"]
    assert [c["name"] for c in rule["secondary"]] == ["S1", "S2", "S3"]
    first = rule["secondary"][0]
    assert first["ours"] == "aucreshape_6x"
    assert first["minus"] == "matched_above_6x"
    assert first["delta"] > 0.0
    assert [c["minus"] for c in rule["secondary"][1:]] == ["weighted_bce", "ce"]
    assert rule["contrasts"][1]["ours"] == "aucreshape"
    assert rule["contrasts"][2]["ours"] == "aucreshape"


def test_secondary_wording_comes_from_the_interval_alone(tmp_path: Path) -> None:
    root = _study(tmp_path / "study", {"aucreshape_6x": 1.5})
    rule = xa.analyse(xa.load_study(root), CODES, n_boot=50)["rules"]["val_sens"]
    for contrast in rule["secondary"]:
        assert contrast["claim"] == reanalysis.claim_label(contrast["lo"], contrast["hi"])


def test_both_controls_are_checked_against_ce(tmp_path: Path) -> None:
    rule = _analyse(_study(tmp_path, {}))["rules"]["val_sens"]
    assert [(c["name"], c["ours"], c["minus"]) for c in rule["checks"]] == [
        ("C1", "matched_random_6x", "ce"),
        ("C2", "matched_above_6x", "ce"),
    ]


def test_the_report_labels_the_secondary_contrasts(tmp_path: Path) -> None:
    root = _study(tmp_path / "study", {})
    text = xa.markdown(xa.analyse(xa.load_study(root), CODES, n_boot=20))
    assert "Secondary" in text
    assert "S1" in text


def test_holm_uses_the_larger_of_the_bootstrap_and_t_p_values(tmp_path: Path) -> None:
    contrast = _analyse(_study(tmp_path, {"aucreshape": 0.5}))["rules"]["val_sens"]["contrasts"][0]
    assert contrast["p"] == max(contrast["p_bootstrap"], contrast["p_t"])


def test_integrity_finds_missing_duplicate_and_mixed_code_runs(tmp_path: Path) -> None:
    root = tmp_path / "study"
    _write_run(root, "arm-1", 2000, 0.0)
    _write_run(root, "arm-1", 2000, 0.0)
    _write_run(root, "arm-2", 2000, 0.0)
    study = xa.load_study(root)
    study[0]["code"], study[1]["code"], study[2]["code"] = "aaa", "aaa", "bbb"
    report = xa.integrity(study, expected=["pc_arm-1_2000", "pc_arm-2_2000", "pc_arm-3_2000"])
    assert report["missing"] == ["pc_arm-3_2000"]
    assert report["duplicates"] == ["pc_arm-1_2000"]
    assert report["code_hashes"] == ["aaa", "bbb"]


def test_a_rerun_cannot_silently_replace_a_run(tmp_path: Path) -> None:
    _write_run(tmp_path, "arm-1", 2000, 0.0)
    _write_run(tmp_path, "arm-1", 2000, 0.0)
    _write_run(tmp_path, "arm-2", 2000, 0.0)
    with pytest.raises(ValueError, match="more than once"):
        xa.analyse(xa.load_study(tmp_path), {"ce": "arm-1", "aucreshape": "arm-2"}, n_boot=2)


def test_an_arm_with_no_finished_run_is_reported_not_a_crash(tmp_path: Path) -> None:
    root = tmp_path / "study"
    for arm, code in CODES.items():
        for seed in SEEDS:
            failed = arm == "matched_random_6x"
            _write_run(root, code, seed, 0.0, status="failed" if failed else "ok")
    report = _analyse(root)
    rule = report["rules"]["val_sens"]
    assert [c["name"] for c in rule["contrasts"]] == ["P2", "P3"]
    assert rule["unavailable"] == [
        {"name": "P1", "ours": "aucreshape_6x", "minus": "matched_random_6x"}
    ]


def test_each_contrast_lists_the_seeds_that_failed(tmp_path: Path) -> None:
    root = tmp_path / "study"
    for arm, code in CODES.items():
        for seed in SEEDS:
            failed = arm == "aucreshape" and seed == 2001
            _write_run(root, code, seed, 0.0, status="failed" if failed else "ok")
    contrast = _analyse(root)["rules"]["val_sens"]["contrasts"][2]
    assert contrast["failed_seeds"] == [2001]


def test_a_contrast_gives_absolute_sensitivity_beside_the_difference(tmp_path: Path) -> None:
    rule = _analyse(_study(tmp_path, {"aucreshape": 0.5}))["rules"]["val_sens"]
    contrast = rule["contrasts"][2]
    arms = rule["arms"]
    assert contrast["ours_sens"] == pytest.approx(
        np.mean([r["sens_roc"] for r in arms["aucreshape"]])
    )
    assert contrast["theirs_sens"] == pytest.approx(np.mean([r["sens_roc"] for r in arms["ce"]]))
    assert contrast["delta"] == pytest.approx(contrast["ours_sens"] - contrast["theirs_sens"])


def test_the_validation_threshold_readout_has_its_own_contrast(tmp_path: Path) -> None:
    rule = _analyse(_study(tmp_path, {"aucreshape": 0.5}))["rules"]["val_sens"]
    contrast = rule["contrasts"][2]
    at = contrast["at_val_threshold"]
    arms = rule["arms"]
    ours = np.mean([r["sens_at_val_threshold"] for r in arms["aucreshape"]])
    theirs = np.mean([r["sens_at_val_threshold"] for r in arms["ce"]])
    assert at["delta"] == pytest.approx(ours - theirs)
    assert at["lo"] <= at["delta"] <= at["hi"]
    ours_spec = np.mean([r["realised_specificity"] for r in arms["aucreshape"]])
    theirs_spec = np.mean([r["realised_specificity"] for r in arms["ce"]])
    assert at["realised_specificity_delta"] == pytest.approx(ours_spec - theirs_spec)
    assert "claim" not in at


def test_claim_wording_belongs_to_the_primary_rule_only(tmp_path: Path) -> None:
    report = _analyse(_study(tmp_path, {"aucreshape": 0.5}))
    for contrast in report["rules"]["val_sens"]["contrasts"]:
        assert contrast["claim"] in xa.WORDINGS
    for contrast in report["rules"]["val_loss"]["contrasts"]:
        assert contrast["claim"].startswith("sensitivity analysis, not a claim:")


def test_the_control_is_checked_against_ce(tmp_path: Path) -> None:
    rule = _analyse(_study(tmp_path, {"matched_random_6x": 2.0}))["rules"]["val_sens"]
    check = rule["checks"][0]
    assert check["name"] == "C1"
    assert check["ours"] == "matched_random_6x"
    assert check["minus"] == "ce"
    assert check["delta"] > 0.0


def test_no_finding_images_get_their_own_readout(tmp_path: Path) -> None:
    _write_run(tmp_path, "arm-1", 2000, 0.0)
    row = xa.load_study(tmp_path)[0]
    y = row["arrays"]["y_test"]
    row["arrays"]["no_finding_test"] = ((1 - y) * (np.arange(len(y)) % 2)).astype(np.uint8)
    out = xa.read_run(row, "val_sens")
    arrays = row["arrays"]
    keep = (y == 1) | (arrays["no_finding_test"] == 1)
    assert out["sens_no_finding"] == pytest.approx(
        xa.sens_points(arrays["test_2"][keep], y[keep].astype(np.float64))
    )


def test_a_run_without_no_finding_labels_says_so(tmp_path: Path) -> None:
    _write_run(tmp_path, "arm-1", 2000, 0.0)
    assert xa.read_run(xa.load_study(tmp_path)[0], "val_sens")["sens_no_finding"] is None


def test_the_6x_arms_boost_summary_stays_apart_from_the_chosen_setting(tmp_path: Path) -> None:
    _sealed(tmp_path, "arm-2", 2000, "aucreshape", 0.2, None)
    _sealed(tmp_path, "arm-5", 2000, "aucreshape", 0.6, None)
    out = xa.boost_summary(tmp_path, {"aucreshape": "arm-2", "aucreshape_6x": "arm-5"})
    assert out["aucreshape"]["boosted_by_epoch"] == pytest.approx([0.2, 0.4])
    assert out["aucreshape_6x"]["boosted_by_epoch"] == pytest.approx([0.6, 1.2])


def test_a_wide_seed_spread_makes_the_t_p_value_decide_holm(tmp_path: Path) -> None:
    root = tmp_path / "study"
    gains = {2000: 0.2, 2001: 1.5, 2002: 0.6}
    for arm, code in CODES.items():
        for seed, gain in gains.items():
            _write_run(root, code, seed, gain if arm == "aucreshape" else 0.0)
    contrast = xa.analyse(xa.load_study(root), CODES, n_boot=100)["rules"]["val_sens"]["contrasts"][
        2
    ]
    assert contrast["p_t"] > contrast["p_bootstrap"]
    assert contrast["p"] == contrast["p_t"]


def test_each_arm_is_cut_at_its_own_validation_threshold(tmp_path: Path) -> None:
    study = xa.load_study(_study(tmp_path, {}))
    for row in study:
        if row["config"]["code"] == CODES["aucreshape"]:
            for name in row["arrays"]:
                if name.startswith("val_"):
                    row["arrays"][name] = row["arrays"][name] - 1.0
    rule = xa.analyse(study, CODES, n_boot=20)["rules"]["val_sens"]
    at = rule["contrasts"][2]["at_val_threshold"]
    ours = np.mean([r["sens_at_val_threshold"] for r in rule["arms"]["aucreshape"]])
    theirs = np.mean([r["sens_at_val_threshold"] for r in rule["arms"]["ce"]])
    assert abs(ours - theirs) > 1.0
    assert at["delta"] == pytest.approx(ours - theirs)


def test_integrity_reports_the_data_fingerprints(tmp_path: Path) -> None:
    _write_run(tmp_path, "arm-1", 2000, 0.0)
    _write_run(tmp_path, "arm-2", 2000, 0.0)
    study = xa.load_study(tmp_path)
    study[0]["data"], study[1]["data"] = {"a": "x"}, {"a": "y"}
    assert len(xa.integrity(study)["data_fingerprints"]) == 2
