"""The written protocol must state what the code does."""

from __future__ import annotations

from pathlib import Path

from benchmarks.review import xray, xray_analysis

TEXT = (Path(__file__).parents[1] / "PROTOCOL.md").read_text()


def test_the_tuning_grids_in_the_text_are_the_ones_the_code_runs() -> None:
    assert len(xray.AR_TUNING_GRID) == len(xray.BCE_TUNING_GRID) == 4
    for setting, factor in zip(xray.AR_TUNING_GRID, xray.BCE_TUNING_GRID, strict=True):
        row = next(
            line
            for line in TEXT.splitlines()
            if line.startswith("|")
            and f"`{setting['combine']}`, floor $r = {setting['min_ratio']}$" in line
        )
        assert row.rstrip().endswith(f"| {factor:g} |")


def test_the_seed_ranges_in_the_text_are_the_ones_the_code_uses() -> None:
    assert f"pilot {xray.PILOT_SEEDS[0]} up" in TEXT
    assert f"tuning {xray.TUNING_SEEDS[0]} up" in TEXT
    assert f"confirmatory {xray.CONFIRMATORY_START} up" in TEXT
    assert ", ".join(str(s) for s in xray.TUNING_SEEDS[:2]) in TEXT


def test_the_split_seed_and_the_target_are_stated() -> None:
    assert f"split seed {xray.SPLIT_SEED}" in TEXT
    assert f"{round(xray.TARGET * 100)}% specificity" in TEXT


def test_the_paper_boosts_are_stated() -> None:
    boosts = ", ".join(f"{b:g}" for b in xray.PAPER_BOOSTS)
    specs = ", ".join(f"{s:.2f}" for s in xray.PAPER_THRESHOLDS)
    assert f"{boosts} at {specs}" in TEXT


def test_every_permitted_wording_is_in_the_text() -> None:
    assert "gain of at least about 1 point" in TEXT
    assert "gain, size uncertain" in TEXT
    assert "no difference larger than 1 point" in TEXT
    assert "loss, size uncertain" in TEXT
    assert "loss of at least about 1 point" in TEXT
    assert "inconclusive" in TEXT


def test_the_primary_family_is_the_one_analysed() -> None:
    assert [name for name, _, _ in xray_analysis.FAMILY] == ["P1", "P2", "P3"]
    assert xray_analysis.PRIMARY_RULE == "val_sens"
    assert "P1, targeting" in TEXT
    assert "P2, the simple fix" in TEXT
    assert "P3, the baseline" in TEXT


def test_the_text_is_ascii() -> None:
    TEXT.encode("ascii")


def test_the_gentle_extension_in_the_text_is_the_one_the_code_runs() -> None:
    assert xray.AR_GENTLE_EXTENSION == {"combine": "band", "min_ratio": 0.9}
    assert f"`band`, floor $r = {xray.AR_GENTLE_EXTENSION['min_ratio']}$" in TEXT
    assert f"| {xray.BCE_GENTLE_EXTENSION:g} |" in TEXT


def test_the_contrast_setting_in_the_text_is_the_one_the_code_runs() -> None:
    assert xray.CONTRAST_SETTING in xray.AR_TUNING_GRID
    combine, floor = xray.CONTRAST_SETTING["combine"], xray.CONTRAST_SETTING["min_ratio"]
    assert f"`{combine}`, floor $r = {floor}$" in TEXT
    assert [name for name, _, _ in xray_analysis.FAMILY] == ["P1", "P2", "P3"]
    assert [name for name, _, _ in xray_analysis.SECONDARY] == ["S1", "S2", "S3"]
    assert "S1" in TEXT


def test_no_old_wording_or_flag_is_left_in_the_text() -> None:
    assert "small gain" not in TEXT
    assert "small loss" not in TEXT
    assert "--with-paper-setting" not in TEXT
    assert "the paper's setting" not in TEXT.lower().replace("not the paper's setting", "")


def test_the_seed_count_in_the_text_is_the_one_the_code_uses() -> None:
    first, last = xray.CONFIRMATORY_START, xray.CONFIRMATORY_START + xray.CONFIRMATORY_SEEDS - 1
    assert f"**Seed count: {xray.CONFIRMATORY_SEEDS},** seeds {first} to {last}." in TEXT
    assert f"--seeds {xray.CONFIRMATORY_SEEDS}" in TEXT


def test_every_arm_the_code_runs_is_named_in_the_text() -> None:
    assert "| Primary | AUCReshaping at about 6x |" in TEXT
    assert "| Primary | Mass-matched random control |" in TEXT
    assert "| Secondary | Mass-matched control above the threshold |" in TEXT
    assert len(xray.CONFIRMATORY_ARMS) == 6
