"""Tests for benchmarks.review.runner: records, pairing, resume and failures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.review import runner


def tiny_task(nan: bool = False) -> runner.Task:
    gen = torch.Generator().manual_seed(0)

    def part(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = (torch.rand(n, generator=gen) < 0.3).float()
        x = torch.randn(n, 4, generator=gen) + y[:, None]
        return x, y

    (xt, yt), (xv, yv), (xs, ys) = part(600), part(300), part(300)
    if nan:
        xt = xt.clone()
        xt[0, 0] = float("nan")
    return runner.Task("tiny", xt, yt, xv, yv, xs, ys)


def spec(**kw: object) -> runner.RunSpec:
    base = runner.RunSpec(
        task="tiny",
        arm="ce",
        params={},
        seed=0,
        hidden=(8,),
        dropout=0.1,
        batch_size=64,
        epochs=4,
        checkpoints=(2, 4),
        target=0.9,
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def scores(out: Path, row: dict[str, object]) -> dict[str, np.ndarray]:
    with np.load(out / "scores" / f"{row['key']}.npz") as f:
        return dict(f)


def test_a_run_writes_one_row_and_its_scores(tmp_path: Path) -> None:
    row = runner.run(tiny_task(), spec(), tmp_path)
    assert row is not None
    assert row["status"] == "ok"
    assert len(row["history"]) == 4
    assert {"git_sha", "versions", "hardware", "seconds"} <= row.keys()
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["key"] == row["key"]
    s = scores(tmp_path, row)
    assert set(s) == {"val_2", "test_2", "val_4", "test_4"}
    assert s["test_4"].shape == (300,)
    assert s["test_4"].dtype == np.float16
    diag = row["history"][3]["diagnostics"]
    assert 0.0 <= diag["below_train_threshold"] <= 1.0
    assert {"train_threshold", "val_threshold", "test_threshold"} <= diag.keys()


def test_finished_runs_are_skipped(tmp_path: Path) -> None:
    first = runner.run(tiny_task(), spec(), tmp_path)
    again = runner.run(tiny_task(), spec(), tmp_path)
    assert again == first
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 1


def test_boost_zero_reproduces_ce_exactly(tmp_path: Path) -> None:
    # Same seed -> same initialisation and batch order in every arm (paired design).
    ce = runner.run(tiny_task(), spec(), tmp_path)
    ar = runner.run(tiny_task(), spec(arm="aucreshape", params={"boost": 0.0}), tmp_path)
    assert ce is not None
    assert ar is not None
    np.testing.assert_array_equal(scores(tmp_path, ce)["test_4"], scores(tmp_path, ar)["test_4"])


def test_threshold_control_reproduces_aucreshape(tmp_path: Path) -> None:
    params = {"boost": 0.3, "min_negatives": 20}
    ar = runner.run(tiny_task(), spec(arm="aucreshape", params=params), tmp_path)
    ctl = runner.run(tiny_task(), spec(arm="threshold_control", params=params), tmp_path)
    assert ar is not None
    assert ctl is not None
    np.testing.assert_allclose(
        scores(tmp_path, ar)["test_4"], scores(tmp_path, ctl)["test_4"], atol=1e-3
    )
    assert ar["history"][-1]["boosted_fraction"] > 0


def test_interrupted_run_resumes_to_the_same_result(tmp_path: Path) -> None:
    whole = runner.run(tiny_task(), spec(arm="aucreshape", params={"boost": 0.3}), tmp_path / "a")
    part = runner.run(
        tiny_task(),
        spec(arm="aucreshape", params={"boost": 0.3}),
        tmp_path / "b",
        max_epochs=3,
    )
    assert part is None  # stopped after 3 of 4 epochs, state saved
    rest = runner.run(tiny_task(), spec(arm="aucreshape", params={"boost": 0.3}), tmp_path / "b")
    assert whole is not None
    assert rest is not None
    np.testing.assert_array_equal(
        scores(tmp_path / "a", whole)["test_4"], scores(tmp_path / "b", rest)["test_4"]
    )
    assert [h["train_loss"] for h in whole["history"]] == pytest.approx(
        [h["train_loss"] for h in rest["history"]]
    )


def test_nan_runs_are_recorded_as_failures(tmp_path: Path) -> None:
    row = runner.run(tiny_task(nan=True), spec(), tmp_path)
    assert row is not None
    assert row["status"].startswith("failed")
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 1


def test_spec_key_depends_on_every_field() -> None:
    assert spec().key() != spec(seed=1).key()
    assert spec().key() != spec(params={"boost": 0.1}).key()
    assert spec().key() == spec().key()


def test_a_passed_deadline_saves_state_and_stops(tmp_path: Path) -> None:
    s = spec(arm="aucreshape", params={"boost": 0.3})
    row = runner.run(tiny_task(), s, tmp_path, deadline=0.0)  # 1970: already passed
    assert row is None
    assert (tmp_path / "state" / f"{s.key()}.pt").exists()
    done = runner.run(tiny_task(), s, tmp_path)
    assert done is not None
    assert len(done["history"]) == 4
    assert not (tmp_path / "state" / f"{s.key()}.pt").exists()
