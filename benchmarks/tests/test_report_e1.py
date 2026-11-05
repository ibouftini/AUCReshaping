"""Smoke test for benchmarks.review.report_e1 on a tiny S-1 sweep with interventions."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmarks.review import e1, report_e1


@pytest.fixture
def tiny_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    kwargs = {"seeds": [0, 1], "tune_seeds": [1000], "scale": 0.02, "epochs": 2, "workers": 1}
    e1.run_experiment("S-1", tmp_path, **kwargs)
    e1.intervene_experiment("S-1", tmp_path, **kwargs)
    monkeypatch.setattr(e1, "OUT", tmp_path)
    monkeypatch.setattr(report_e1, "SCALE", 0.02)
    for cached in (report_e1.rows, report_e1.intervention_rows, report_e1.labels):
        cached.cache_clear()
    yield tmp_path
    for cached in (report_e1.rows, report_e1.intervention_rows, report_e1.labels):
        cached.cache_clear()


def read(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.mark.filterwarnings("ignore:ThresholdTracker.calibrate")  # smoke scale
def test_summaries_cover_every_arm_and_intervention(tiny_sweep: Path) -> None:
    report_e1.summary("S-1")
    report_e1.intervention_summary("S-1")
    final = read(tiny_sweep / "S-1" / "summary.csv")
    assert {r["arm"] for r in final} == set(e1.EVAL_ARMS)
    interventions = read(tiny_sweep / "S-1" / "interventions.csv")
    assert {r["arm"] for r in interventions} == {"ce", *e1.INTERVENTIONS}
    for r in interventions:
        if r["arm"] != "ce":
            assert float(r["lo"]) <= float(r["delta"]) <= float(r["hi"])
            assert r["gains"].endswith("/2")


@pytest.mark.filterwarnings("ignore:ThresholdTracker.calibrate")  # smoke scale
def test_g3_reports_both_criteria(tiny_sweep: Path) -> None:
    out = report_e1.g3()
    assert set(out) >= {"optimum", "best_checkpoint", "selected", "last", "per_epoch",
                        "amended_pass", "original_pass"}  # fmt: skip
    assert set(out["best_checkpoint"]) == set(e1.EVAL_ARMS)
    assert (tiny_sweep / "S-1" / "g3.json").exists()


@pytest.mark.filterwarnings("ignore:ThresholdTracker.calibrate")  # smoke scale
def test_markdown_tables_come_from_the_csvs(
    tiny_sweep: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_e1.summary("S-1")
    report_e1.intervention_summary("S-1")
    report_e1.g3()
    out = tiny_sweep / "S12b_synthetic.md"
    monkeypatch.setattr(report_e1, "REPORT", out)
    report_e1.markdown(["S-1"])
    text = out.read_text()
    assert "## S-1" in text
    assert "| CE (sensitivity) |" in text
    assert "I1: on top of class weights" in text
    final = read(tiny_sweep / "S-1" / "summary.csv")
    weighted = next(r for r in final if r["arm"] == "weighted_bce")
    assert f"{float(weighted['delta']):+.2f}" in text
    assert "Gate G3" in text
