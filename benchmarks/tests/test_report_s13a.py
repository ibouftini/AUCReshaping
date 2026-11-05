"""Audit tables: generated from the result files, never typed by hand."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.review import report_s13a as report


def _row(name: str, tag: str, setting: str, arm: str, delta: float, lo: float, hi: float) -> dict:
    return {
        "name": name,
        "tag": tag,
        "setting": setting,
        "arm": arm,
        "delta": delta,
        "lo": lo,
        "hi": hi,
        "paired_sd": 0.5,
        "seeds": 10,
        "status": "ok",
        "claim": "inconclusive",
    }


def test_interval_uses_signed_two_decimals() -> None:
    assert report.interval(0.3867, -0.0342, 0.7948) == "+0.39 [-0.03, +0.79]"


def test_load_contrasts_merges_kernels_and_keeps_failures(tmp_path: Path) -> None:
    for i in (0, 1):
        folder = tmp_path / f"kernel-{i}" / "results"
        folder.mkdir(parents=True)
        row = _row("S-3", str(i), "prob", "placebo", 0.1, 0.0, 0.2)
        if i == 1:
            row = {
                "name": "S-3",
                "tag": "1",
                "setting": "prob",
                "arm": "placebo",
                "status": "error",
            }
        (folder / "contrasts.jsonl").write_text(json.dumps(row) + "\n")
    rows = report.load_contrasts(tmp_path)
    assert len(rows) == 2
    assert sorted(r["status"] for r in rows) == ["error", "ok"]


def test_claim_counts_separate_settings_and_arms() -> None:
    rows = [
        {**_row("S-3", "0", "prob", "random_boost", 0, -1, 1), "claim": "inconclusive"},
        {**_row("S-3", "1", "prob", "random_boost", 0, -1, 1), "claim": "inconclusive"},
        {
            **_row("S-3", "0", "weight", "random_boost", 0, -1, 1),
            "claim": "gain, size uncertain",
        },
    ]
    counts = report.claim_counts(rows)
    assert counts[("prob", "random_boost", "inconclusive")] == 2
    assert counts[("weight", "random_boost", "gain, size uncertain")] == 1
    assert ("weight", "random_boost", "inconclusive") not in counts


def test_scenario_table_has_one_row_per_level_and_one_column_per_arm() -> None:
    rows = [
        _row("S-3", "0.0", "prob", "random_boost", 0.2, -0.1, 0.5),
        _row("S-3", "0.0", "prob", "weighted_bce", -1.5, -2.0, -1.1),
        _row("S-3", "0.01", "prob", "random_boost", 0.1, -0.2, 0.4),
    ]
    lines = report.scenario_table(rows, "S-3", "prob", ("random_boost", "weighted_bce"))
    assert lines[0] == "| level | random_boost | weighted_bce |"
    assert len(lines) == 4
    assert "+0.20 [-0.10, +0.50]" in lines[2]
    assert "-1.50 [-2.00, -1.10]" in lines[2]
    assert lines[3].endswith("| n/a |")


def test_failed_contrasts_are_listed_not_dropped() -> None:
    rows = [{"name": "S-3", "tag": "0.0", "setting": "prob", "arm": "placebo", "status": "error"}]
    lines = report.scenario_table(rows, "S-3", "prob", ("placebo",))
    assert lines[2].endswith("| failed |")


def test_seeds_needed_uses_the_t_distribution() -> None:
    # n = 7: 2.447 * 1 / sqrt(7) = 0.925; n = 6: 2.571 * 1 / sqrt(6) = 1.050.
    assert report.seeds_needed(1.0) == 7
    # n = 11: 2.228 * 1.4 / sqrt(11) = 0.941; n = 10: 2.262 * 1.4 / sqrt(10) = 1.002.
    assert report.seeds_needed(1.4) == 11
    assert report.seeds_needed(0.1) == 2


def test_seeds_needed_scales_with_the_half_width() -> None:
    assert report.seeds_needed(1.0, half_width=2.0) == report.seeds_needed(0.5)


def test_unique_cells_counts_a_shared_experiment_once() -> None:
    first = {**_row("S-3", "0.02", "prob", "placebo", 0.1, 0, 1), "per_seed": [1.0, 2.0]}
    copy = {**_row("S-5", "0.0", "prob", "placebo", 0.1, 0, 1), "per_seed": [1.0, 2.0]}
    other = {**_row("S-3", "0.05", "prob", "placebo", 0.1, 0, 1), "per_seed": [1.0, 3.0]}
    rows = report.unique_cells([first, copy, other])
    assert len(rows) == 2
    assert {r["tag"] for r in rows} == {"0.02", "0.05"}


def test_generic_table_rounds_numbers_and_lists() -> None:
    lines = report.generic_table(
        [{"arm": "ce", "value": "0.12345", "per_seed": "[1.2345, 2.3456]"}],
        ("arm", "value", "per_seed"),
        digits=3,
    )
    assert lines[0] == "| arm | value | per_seed |"
    assert lines[2] == "| ce | 0.123 | 1.234 / 2.346 |"


def test_seed_spread_summarises_each_setting_and_arm() -> None:
    rows = [
        {
            **_row("S-3", "0.0", "prob", "random_boost", 0, -1, 1),
            "paired_sd": 1.0,
            "per_seed": [1.0],
        },
        {
            **_row("S-3", "0.1", "prob", "random_boost", 0, -1, 1),
            "paired_sd": 3.0,
            "per_seed": [2.0],
        },
    ]
    spread = report.seed_spread(rows)
    assert len(spread) == 1
    assert spread[0]["setting"] == "prob"
    assert spread[0]["median_sd"] == 2.0
    assert spread[0]["max_sd"] == 3.0
    assert spread[0]["seeds_for_median"] == report.seeds_needed(2.0)


def test_contrasts_saved_with_the_old_wording_are_read_in_the_new_wording(tmp_path: Path) -> None:
    (tmp_path / "contrasts.jsonl").write_text(
        json.dumps({"status": "ok", "claim": "small loss, size uncertain"}) + "\n"
    )
    assert report.load_contrasts(tmp_path)[0]["claim"] == "loss, size uncertain"
