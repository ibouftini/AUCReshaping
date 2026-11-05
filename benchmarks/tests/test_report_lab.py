"""Tables of the fine-tuning lab, computed from its result rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.review import report_lab

NEAR = "no difference larger than 1 point"
GAIN = "gain of at least about 1 point"


def _row(
    name: str,
    tag: str,
    regime: str = "rare",
    ce: tuple[float, ...] = (10.0, 12.0),
    ar: tuple[float, ...] = (11.0, 13.0),
    claim: str = NEAR,
    chosen: dict[str, Any] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    delta = ar[0] - ce[0]
    return {
        "name": name,
        "tag": tag,
        "regime": regime,
        "status": status,
        "chosen": chosen or {},
        "arms": {
            "ce": list(ce),
            "aucreshape": list(ar),
            "weighted_bce": [c + 2.0 for c in ce],
        },
        "contrasts": [
            {"minus": "ce", "delta": delta, "lo": delta - 0.5, "hi": delta + 0.5, "claim": claim},
        ],
        "diagnostics": {
            "strength": {"mass_ratio": 3.0},
            "recipient_overlap": {"matched_random": 0.5},
        },
    }


def test_a_level_shared_by_two_scenarios_counts_once() -> None:
    rows = [_row("S-3", "0.02"), _row("S-5", "0.0"), _row("S-4", "0.1", ce=(1.0, 2.0))]
    kept = report_lab.unique_levels(rows)
    assert [(r["name"], r["tag"]) for r in kept] == [("S-3", "0.02"), ("S-4", "0.1")]


def test_the_same_level_in_another_regime_is_another_experiment() -> None:
    rows = [_row("S-3", "0.02"), _row("S-3", "0.02", regime="paper")]
    assert len(report_lab.unique_levels(rows)) == 2


def test_a_failed_level_is_counted_and_not_analysed() -> None:
    rows = [
        _row("S-3", "0.02"),
        {"name": "S-4", "tag": "0.1", "regime": "rare", "status": "failed"},
    ]
    assert len(report_lab.failed(rows)) == 1
    assert len(report_lab.unique_levels(rows)) == 1


def test_claims_are_counted_per_contrast_and_regime() -> None:
    rows = [
        _row("S-3", "0.02", claim=GAIN),
        _row("S-4", "0.1", ce=(1.0, 2.0), claim=NEAR),
        _row("S-4", "0.2", ce=(3.0, 4.0), regime="paper", claim=NEAR),
    ]
    counts = report_lab.claim_counts(rows, "rare")
    assert counts["ce"][GAIN] == 1
    assert counts["ce"][NEAR] == 1
    assert sum(report_lab.claim_counts(rows, "paper")["ce"].values()) == 1


def test_median_gain_over_ce_is_of_level_means() -> None:
    rows = [
        _row("S-3", "0.02", ce=(10.0, 12.0), ar=(11.0, 13.0)),
        _row("S-4", "0.1", ce=(1.0, 2.0), ar=(4.0, 5.0)),
        _row("S-4", "0.2", ce=(3.0, 4.0), ar=(8.0, 9.0)),
    ]
    medians = report_lab.median_gain_over_ce(rows, "rare")
    assert medians["aucreshape"] == pytest.approx(3.0)
    assert medians["weighted_bce"] == pytest.approx(2.0)
    assert "ce" not in medians


def test_grid_edges_are_counted_at_both_ends() -> None:
    grid = ({"r": 0.75}, {"r": 0.5}, {"r": 0.1})
    rows = [
        _row("S-3", "0.02", chosen={"aucreshape": {"r": 0.75}}),
        _row("S-4", "0.1", ce=(1.0, 2.0), chosen={"aucreshape": {"r": 0.1}}),
        _row("S-4", "0.2", ce=(3.0, 4.0), chosen={"aucreshape": {"r": 0.1}}),
        _row("S-4", "0.05", ce=(5.0, 6.0), chosen={"aucreshape": {"r": 0.5}}),
    ]
    assert report_lab.edge_counts(rows, "rare", "aucreshape", grid) == (1, 2)


def test_factors_are_read_from_the_weighted_arm() -> None:
    rows = [_row("S-3", "0.02", chosen={"weighted_bce": {"factor": 16.0}})]
    grid = tuple({"factor": f} for f in (1.25, 16.0))
    assert report_lab.edge_counts(rows, "rare", "weighted_bce", grid) == (0, 1)


def test_the_report_names_every_regime_and_is_ascii(tmp_path: Path) -> None:
    folder = tmp_path / "kernel-0" / "results"
    folder.mkdir(parents=True)
    lines = [json.dumps(_row("S-3", "0.02")), json.dumps(_row("S-3", "0.02", regime="paper"))]
    (folder / "levels.jsonl").write_text("\n".join(lines) + "\n")
    rows = report_lab.load_levels(tmp_path)
    assert len(rows) == 2
    text = report_lab.markdown(rows)
    assert "rare" in text
    assert "paper" in text
    text.encode("ascii")


def test_results_saved_with_the_old_wording_are_read_in_the_new_wording(tmp_path: Path) -> None:
    folder = tmp_path / "kernel-0" / "results"
    folder.mkdir(parents=True)
    row = _row("S-3", "0.02", claim="small gain, size uncertain")
    (folder / "levels.jsonl").write_text(json.dumps(row) + "\n")
    rows = report_lab.load_levels(tmp_path)
    assert report_lab.claim_counts(rows, "rare")["ce"]["gain, size uncertain"] == 1
