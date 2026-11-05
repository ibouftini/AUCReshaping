"""The credit-card baseline reconciliation (exploratory, CE only)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from benchmarks.review import cc_reconcile as cc

USED_SEEDS = set(range(10)) | set(range(100, 105)) | set(range(1000, 1003))


def _fraud(n: int = 2000, frauds: int = 50) -> np.ndarray:
    fraud = np.zeros(n, dtype=bool)
    fraud[:frauds] = True
    return fraud


def test_paper_split_defaults_match_table_2() -> None:
    train, test = cc.paper_split(_fraud(284_807, 492), split_seed=1500)
    fraud = _fraud(284_807, 492)
    assert len(test) == 56_962
    assert len(train) == 227_845  # Table 2 prints 227,885, which exceeds the dataset by 40
    assert fraud[test].sum() == 101
    assert fraud[train].sum() == 391


def test_paper_split_partitions_the_data() -> None:
    train, test = cc.paper_split(_fraud(), 7, n_test=400, n_test_fraud=10)
    assert len(set(train) & set(test)) == 0
    assert sorted([*train, *test]) == list(range(2000))


def test_paper_split_depends_only_on_the_seed() -> None:
    a = cc.paper_split(_fraud(), 7, n_test=400, n_test_fraud=10)
    b = cc.paper_split(_fraud(), 7, n_test=400, n_test_fraud=10)
    c = cc.paper_split(_fraud(), 8, n_test=400, n_test_fraud=10)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_standardise_uses_training_statistics_only() -> None:
    train = np.array([[0.0], [2.0]], dtype=np.float32)
    test = np.array([[1000.0]], dtype=np.float32)
    out_train, out_test = cc.standardise(train, test)
    assert out_train.mean() == pytest.approx(0.0, abs=1e-6)
    assert out_test[0, 0] == pytest.approx((1000.0 - 1.0) / (1.0 + 1e-6), rel=1e-5)


def test_loss_on_normals_equals_pos_weight_in_the_paper_direction() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    normal = torch.tensor([1.0, 1.0, 0.0, 0.0])
    expected = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))(logits, normal)
    assert float(cc.weighted_bce(logits, normal, 5.0, "normal")) == pytest.approx(float(expected))


def test_loss_on_frauds_up_weights_frauds() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    normal = torch.tensor([1.0, 1.0, 0.0, 0.0])
    # Weighting frauds is pos_weight on the mirrored problem.
    expected = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))(-logits, 1 - normal)
    assert float(cc.weighted_bce(logits, normal, 5.0, "fraud")) == pytest.approx(float(expected))


def test_loss_rejects_an_unknown_class() -> None:
    with pytest.raises(ValueError, match="weighted_class"):
        cc.weighted_bce(torch.zeros(1), torch.ones(1), 5.0, "both")


def _toy(n: int = 600) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    normal = (torch.rand(n, generator=gen) < 0.8).float()
    x = torch.randn(n, 4, generator=gen) + 2.0 * normal.unsqueeze(1)
    return x[:400], normal[:400], x[400:]


FAST = replace(cc.PAPER, epochs=4, batch_size=32, widths=(8,), checkpoints=(2, 4))


def test_run_is_reproducible_and_keeps_the_checkpoints() -> None:
    x, y, x_test = _toy()
    a = cc.train_run(x, y, x_test, FAST, seed=3)
    b = cc.train_run(x, y, x_test, FAST, seed=3)
    c = cc.train_run(x, y, x_test, FAST, seed=4)
    assert a.status == "ok"
    assert sorted(a.scores) == [2, 4]
    assert a.scores[4].shape == (200,)
    assert a.scores[4].dtype == np.float32
    assert np.array_equal(a.scores[4], b.scores[4])
    assert not np.array_equal(a.scores[4], c.scores[4])
    assert len(a.train_loss) == 4


@pytest.mark.parametrize("weighted", ["normal", "fraud"])
def test_scores_rank_normals_high_whichever_class_is_weighted(weighted: str) -> None:
    x, y, x_test = _toy()
    gen = torch.Generator().manual_seed(0)
    normal = (torch.rand(600, generator=gen) < 0.8).float()
    y_test = normal[400:].numpy().astype(bool)
    config = replace(FAST, epochs=30, checkpoints=(30,), weighted_class=weighted, dropout=0.0)
    scores = cc.train_run(x, y, x_test, config, seed=0).scores[30]
    assert scores[y_test].mean() > scores[~y_test].mean()


def test_a_diverging_run_is_recorded_not_raised() -> None:
    x, y, x_test = _toy()
    config = replace(FAST, epochs=50, checkpoints=(50,), lr=1e12, momentum=0.9)
    result = cc.train_run(x, y, x_test, config, seed=0)
    assert result.status == "nonfinite"
    assert len(result.train_loss) < 50
    assert result.scores == {}


def test_plan_is_the_declared_design() -> None:
    jobs = cc.jobs()
    assert len(jobs) == 60
    assert len({(j.config.name, j.split_seed, j.seed) for j in jobs}) == 60
    names = {j.config.name for j in jobs}
    assert names == {"paper", "pw_fraud", "momentum_0", "lr_1e-3", "lr_1e-1", "wide", "raw"}
    # Factors share one split; only the paper configuration is repeated over splits.
    assert {j.split_seed for j in jobs if j.config.name != "paper"} == {cc.BASE_SPLIT}
    assert len({j.split_seed for j in jobs if j.config.name == "paper"}) == 6
    # Fresh seeds: not used before, and clear of the confirmatory range.
    used = {j.seed for j in jobs} | {j.split_seed for j in jobs}
    assert not used & USED_SEEDS
    assert max(used) < 2000


def test_factors_change_exactly_one_setting() -> None:
    base = cc.PAPER
    for job in cc.jobs():
        config = job.config
        changed = [
            field
            for field in (
                "widths",
                "dropout",
                "lr",
                "momentum",
                "pos_weight",
                "weighted_class",
                "standardise",
                "batch_size",
                "epochs",
            )
            if getattr(config, field) != getattr(base, field)
        ]
        assert len(changed) == (0 if config.name == "paper" else 1), config.name


def test_paper_configuration_states_what_the_paper_states() -> None:
    config = cc.PAPER
    assert (config.dropout, config.batch_size, config.pos_weight) == (0.5, 100, 5.0)
    assert config.epochs == 200


def test_shards_partition_the_jobs() -> None:
    jobs = cc.jobs()
    shards = [cc.shard(jobs, k, 5) for k in range(5)]
    assert sum(len(s) for s in shards) == len(jobs)
    assert {len(s) for s in shards} == {12}
    keys = [j.key for s in shards for j in s]
    assert len(set(keys)) == len(jobs)


def test_kernel_is_private_cpu_and_offline(tmp_path: Path) -> None:
    folder = cc.make_kernel(tmp_path, user="someone", index=2, shards=5)
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/aucreshape-s13a-cc-2"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is False
    assert meta["dataset_sources"] == ["mlg-ulb/creditcardfraud"]
    code = (folder / meta["code_file"]).read_text()
    compile(code, "kernel", "exec")
    assert "\nKERNEL_SHARD: int | None = 2\n" in code
    assert "\nKERNEL_SHARDS: int = 5\n" in code


def test_smoke_kernel_limits_jobs_and_epochs(tmp_path: Path) -> None:
    folder = cc.make_kernel(tmp_path, user="someone", index=0, shards=5, smoke=True)
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/aucreshape-s13a-cc-smoke"
    code = (folder / meta["code_file"]).read_text()
    assert "\nKERNEL_EPOCHS: int | None = 2\n" in code
    assert "\nKERNEL_MAX_JOBS: int | None = 4\n" in code


def _write_run(
    root: Path, key: str, name: str, scores: np.ndarray, y: np.ndarray, status: str = "ok"
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "key": key,
        "status": status,
        "config": {"name": name},
        "split_seed": 1500,
        "seed": int(key[-1]),
        "seconds": 1.0,
    }
    with (root / "runs.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    if status == "ok":
        np.savez_compressed(root / f"{key}.npz", test_200=scores, y_normal=y)


def test_summary_counts_failed_runs_and_uses_the_paper_direction(tmp_path: Path) -> None:
    y = np.array([True] * 8 + [False] * 2)
    perfect = np.array([*np.arange(10.0, 18.0), 0.0, 1.0], dtype=np.float32)
    _write_run(tmp_path, "paper_0", "paper", perfect, y)
    _write_run(tmp_path, "paper_1", "paper", perfect, y)
    _write_run(tmp_path, "paper_2", "paper", perfect, y, status="nonfinite")
    rows = cc.summarise(tmp_path, epoch=200)
    paper = next(r for r in rows if r["config"] == "paper")
    assert paper["runs"] == 3
    assert paper["failed"] == 1
    assert paper["auc_mean"] == pytest.approx(1.0)
    assert paper["sens@0.98_mean"] == pytest.approx(1.0)
    assert paper["sens@0.98_per_run"] == [pytest.approx(1.0), pytest.approx(1.0)]


def test_extension_repeats_only_momentum_0_on_the_other_splits() -> None:
    plan = cc.extension_jobs()
    assert len(plan) == 25
    assert {j.config.name for j in plan} == {"momentum_0"}
    assert {j.split_seed for j in plan} == set(cc.REPEAT_SPLITS)
    assert {j.seed for j in plan} == set(cc.SEEDS)
    # It adds to the declared design without changing it or repeating a run.
    assert not {j.key for j in plan} & {j.key for j in cc.jobs()}
    assert len(cc.jobs()) == 60


def test_extension_kernel_runs_the_extension(tmp_path: Path) -> None:
    folder = cc.make_kernel(tmp_path, user="someone", index=1, shards=2, extension=True)
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/aucreshape-s13a-cc-ext-1"
    assert meta["is_private"] is True
    code = (folder / meta["code_file"]).read_text()
    compile(code, "kernel", "exec")
    assert "\nKERNEL_EXTENSION: bool = True\n" in code
