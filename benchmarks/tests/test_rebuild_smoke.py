"""One command to rebuild every table, and one smoke run of each pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.review import rebuild, smoke


def test_rebuild_runs_steps_in_order_and_skips_those_without_inputs(tmp_path: Path) -> None:
    present = tmp_path / "present"
    present.mkdir()
    calls: list[str] = []
    steps = [
        rebuild.Step("first", (present,), lambda: calls.append("first")),
        rebuild.Step("missing", (tmp_path / "absent",), lambda: calls.append("missing")),
        rebuild.Step("second", (), lambda: calls.append("second")),
    ]
    report = rebuild.run(steps)
    assert calls == ["first", "second"]
    assert [(r["step"], r["status"]) for r in report] == [
        ("first", "rebuilt"),
        ("missing", "skipped"),
        ("second", "rebuilt"),
    ]
    assert str(tmp_path / "absent") in report[1]["detail"]


def test_rebuild_reports_a_failed_step_and_keeps_going(tmp_path: Path) -> None:
    def broken() -> None:
        raise RuntimeError("no such column")

    report = rebuild.run(
        [rebuild.Step("broken", (), broken), rebuild.Step("next", (), lambda: None)]
    )
    assert report[0]["status"] == "failed"
    assert "no such column" in report[0]["detail"]
    assert report[1]["status"] == "rebuilt"


def test_rebuild_exit_code_counts_failures_not_skips() -> None:
    assert rebuild.exit_code([{"status": "rebuilt"}, {"status": "skipped"}]) == 0
    assert rebuild.exit_code([{"status": "failed"}]) == 1


def test_the_real_steps_never_train() -> None:
    names = " ".join(step.name for step in rebuild.steps(Path("results"), Path("out")))
    assert "tables" in names
    for step in rebuild.steps(Path("results"), Path("out")):
        assert "train" not in step.name


@pytest.mark.parametrize("pipeline", sorted(smoke.PIPELINES))
def test_each_pipeline_smoke_run_passes(pipeline: str, tmp_path: Path) -> None:
    smoke.PIPELINES[pipeline](tmp_path)


def test_smoke_reports_every_pipeline(tmp_path: Path) -> None:
    def broken(_: Path) -> None:
        raise RuntimeError("boom")

    report = smoke.run({"good": lambda _: None, "bad": broken}, tmp_path)
    assert {r["pipeline"]: r["status"] for r in report} == {"good": "ok", "bad": "failed"}


def test_steps_that_read_scores_need_the_scores_not_just_their_folder() -> None:
    needs = {step.name: step.needs for step in rebuild.steps(Path("results"), Path("out"))}
    assert any(
        "scores" in path.parts for path in needs["synthetic validation threshold and noise floor"]
    )
    assert any(path.suffix == ".npz" for path in needs["credit-card baseline"])
