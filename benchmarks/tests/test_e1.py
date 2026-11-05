"""Tests for benchmarks.review.e1, the synthetic-laboratory driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.review import e1


@pytest.mark.parametrize("name", sorted(e1.EXPERIMENTS))
def test_every_level_builds_a_task(name: str) -> None:
    exp = e1.EXPERIMENTS[name]
    for level in exp.levels:
        task = exp.task(level, seed=0, scale=0.02)
        assert len(task.y_train) > 0
        assert int((task.y_train > 0.5).sum()) > 0
        assert task.oracle is not None
    if name == "S-5":
        assert "flipped" in exp.task(exp.levels[-1], seed=0, scale=0.02).meta


def test_choose_picks_the_best_mean_selected_validation_sensitivity() -> None:
    def row(params: dict[str, float], seed: int, best: float) -> dict[str, object]:
        history = [{"epoch": 1, "val_sens": best - 0.1}, {"epoch": 2, "val_sens": best}]
        return {"spec": {"arm": "focal", "params": params, "seed": seed, "checkpoints": [1, 2]},
                "status": "ok", "history": history}  # fmt: skip

    rows = [row({"gamma": 1.0}, s, 0.5) for s in (1000, 1001)]
    rows += [row({"gamma": 2.0}, 1000, 0.9), row({"gamma": 2.0}, 1001, 0.2)]
    assert e1.choose(rows, "focal") == {"gamma": 2.0}  # mean 0.55 beats 0.50
    rows.append(row({"gamma": 3.0}, 1000, 0.7))
    rows.append(row({"gamma": 3.0}, 1001, 0.7))
    assert e1.choose(rows, "focal") == {"gamma": 3.0}


# The smoke scale leaves 400 validation samples, so the held-out arm correctly warns that
# 0.98 specificity needs 500 negatives; at full scale validation has ~19,000.
@pytest.mark.filterwarnings("ignore:ThresholdTracker.calibrate")
def test_smoke_run_writes_every_arm(tmp_path: Path) -> None:
    e1.run_experiment("S-1", tmp_path, seeds=[0], tune_seeds=[1000], scale=0.02, epochs=2,
                      workers=1)  # fmt: skip
    rows = [
        json.loads(line)
        for p in tmp_path.rglob("runs.jsonl")
        for line in p.read_text().splitlines()
    ]
    arms = {r["spec"]["arm"] for r in rows if r["spec"]["seed"] == 0}
    assert arms == set(e1.EVAL_ARMS)
    assert all(r["status"] == "ok" for r in rows)
    chosen = json.loads((tmp_path / "S-1" / "chosen.json").read_text())
    assert set(chosen) == set(e1.TUNED)


@pytest.mark.filterwarnings("ignore:ThresholdTracker.calibrate")  # smoke scale, see above
def test_extension_retunes_arms_that_hit_the_grid_edge(tmp_path: Path) -> None:
    kwargs = {"seeds": [0], "tune_seeds": [1000], "scale": 0.02, "epochs": 2, "workers": 1}
    e1.run_experiment("S-1", tmp_path, **kwargs)
    edge = {"aucreshape": e1.TUNED["aucreshape"][-1], "weighted_bce": e1.TUNED["weighted_bce"][0],
            "focal": e1.TUNED["focal"][0]}  # fmt: skip
    (tmp_path / "S-1" / "chosen.json").write_text(json.dumps(edge))
    chosen = e1.extend_experiment("S-1", tmp_path, **kwargs)
    assert json.loads((tmp_path / "S-1" / "chosen_original.json").read_text()) == edge
    assert chosen["aucreshape"] in e1.TUNED["aucreshape"] + e1.EXTRA["aucreshape"]
    assert chosen["weighted_bce"] == edge["weighted_bce"]  # not at the edge: untouched
    tuned = [json.loads(line) for p in (tmp_path / "S-1" / "tune").rglob("runs.jsonl")
             for line in p.read_text().splitlines()]  # fmt: skip
    assert {"mode": "weight", "boost": 16.0} in [r["spec"]["params"] for r in tuned]
    evaluated = [r["spec"]["params"] for r in e1.current_rows(tmp_path / "S-1")
                 if r["spec"]["arm"] == "random_boost"]  # fmt: skip
    assert evaluated == [chosen["aucreshape"]]


def test_expected_params_follow_the_chosen_settings() -> None:
    chosen = {"aucreshape": {"mode": "weight", "boost": 4.0}, "weighted_bce": {"pos_weight": 8.0},
              "focal": {"gamma": 1.0}}  # fmt: skip
    expected = e1.expected_params(chosen)
    assert expected["ce"] == {}
    assert expected["placebo"] == {"mode": "weight", "boost": 4.0}
    assert expected["weighted_bce"] == {"pos_weight": 8.0}


@pytest.mark.filterwarnings("ignore:ThresholdTracker.calibrate")  # smoke scale, see above
def test_interventions_run_every_arm(tmp_path: Path) -> None:
    kwargs = {"seeds": [0], "tune_seeds": [1000], "scale": 0.02, "epochs": 2, "workers": 1}
    e1.run_experiment("S-1", tmp_path, **kwargs)
    chosen = e1.intervene_experiment("S-1", tmp_path, **kwargs)
    assert set(chosen) == set(e1.INTERVENTION_GRIDS)
    rows = [json.loads(line) for p in (tmp_path / "S-1" / "interventions").rglob("runs.jsonl")
            if "tune" not in p.parts for line in p.read_text().splitlines()]  # fmt: skip
    assert {r["spec"]["arm"] for r in rows} == set(e1.INTERVENTIONS)
    warm = next(r for r in rows if r["spec"]["arm"] == "warm_aucreshape")
    assert warm["spec"]["params"]["warmup_steps"] >= 1  # resolved from the 50% fraction
    assert all(r["status"] == "ok" for r in rows)


@pytest.mark.filterwarnings("ignore:ThresholdTracker.calibrate")  # smoke scale, see above
def test_current_rows_skip_interventions_and_intervention_rows_find_them(tmp_path: Path) -> None:
    # Regression: intervention rows sit under the scenario folder too, and their arms are not
    # in expected_params, so current_rows raised KeyError once I1-I5 had run.
    kwargs = {"seeds": [0], "tune_seeds": [1000], "scale": 0.02, "epochs": 2, "workers": 1}
    e1.run_experiment("S-1", tmp_path, **kwargs)
    e1.intervene_experiment("S-1", tmp_path, **kwargs)
    current = e1.current_rows(tmp_path / "S-1")
    assert {r["spec"]["arm"] for r in current} == set(e1.EVAL_ARMS)
    found = e1.intervention_rows(tmp_path / "S-1")
    assert {r["spec"]["arm"] for r in found} == set(e1.INTERVENTIONS)
    assert all("tune" not in Path(r["_dir"]).parts for r in found)


@pytest.mark.filterwarnings("ignore:ThresholdTracker.calibrate")  # smoke scale, see above
def test_current_rows_can_read_the_original_choice_after_an_extension(tmp_path: Path) -> None:
    kwargs = {"seeds": [0], "tune_seeds": [1000], "scale": 0.02, "epochs": 2, "workers": 1}
    e1.run_experiment("S-1", tmp_path, **kwargs)
    edge = {"aucreshape": e1.TUNED["aucreshape"][-1], "weighted_bce": e1.TUNED["weighted_bce"][0],
            "focal": e1.TUNED["focal"][0]}  # fmt: skip
    level = e1.EXPERIMENTS["S-1"].centre
    folder = tmp_path / "S-1" / e1.level_tag(level) / "seed0"
    arms = list(e1.expected_params(edge).items())
    e1._job(("S-1", level, 0, arms, 0.02, 2, str(folder)))  # evaluation at the edge setting
    (tmp_path / "S-1" / "chosen.json").write_text(json.dumps(edge))
    chosen = e1.extend_experiment("S-1", tmp_path, **kwargs)
    if chosen["aucreshape"] == edge["aucreshape"]:
        pytest.skip("smoke-scale tuning kept the edge setting; nothing to tell apart")
    original = e1.current_rows(tmp_path / "S-1", "chosen_original.json")
    final = e1.current_rows(tmp_path / "S-1")
    pick = lambda rows: [r["spec"]["params"] for r in rows if r["spec"]["arm"] == "aucreshape"]  # noqa: E731
    assert pick(original) == [edge["aucreshape"]]
    assert pick(final) == [chosen["aucreshape"]]
