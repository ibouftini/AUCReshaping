"""Promises the evaluation makes to a sceptical reader, pinned by tests.

Already covered elsewhere, and listed so this file is the index:
- the metric against scikit-learn: ``tests/unit/test_metrics.py``;
- the fast metric against the library: ``test_analysis.py``;
- no patient in two splits: ``test_replication.py::test_load_nih_validation_split_is_patient_wise``;
- the random control with the shuffle off: ``test_replication.py::
  test_random_boost_without_shuffle_reproduces_aucreshaping``.

The new chest X-ray pipeline must pass the same promises when it is written.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from benchmarks import replication
from benchmarks.review import analysis, e1, runner


class ValidationOnly(dict):
    """A history entry that refuses to give up anything but validation data."""

    def __getitem__(self, key: str) -> Any:
        if "test" in key:
            raise AssertionError(f"selection read test data: {key}")
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if "test" in key:
            raise AssertionError(f"selection read test data: {key}")
        return super().get(key, default)

    def items(self) -> Iterator[tuple[str, Any]]:  # type: ignore[override]
        raise AssertionError("selection walked the whole record")


def _history() -> list[dict[str, Any]]:
    return [
        ValidationOnly(epoch=e, val_sens=s, val_loss=loss, test_sens=0.99)
        for e, s, loss in ((10, 0.2, 0.5), (20, 0.4, 0.3), (50, 0.3, 0.1))
    ]


@pytest.mark.parametrize(("rule", "epoch"), [("val", 20), ("loss", 50), ("last", 50)])
def test_selected_epoch_reads_validation_only(rule: str, epoch: int) -> None:
    row = {"spec": {"checkpoints": [10, 20, 50]}, "history": _history()}
    assert analysis.selected_epoch(row, rule) == epoch


def test_tuning_choice_reads_validation_only() -> None:
    rows = [
        {
            "spec": {"arm": "aucreshape", "params": {"boost": b}, "checkpoints": [10, 20, 50]},
            "status": "ok",
            "history": [ValidationOnly(epoch=10, val_sens=b, test_sens=1 - b)],
        }
        for b in (0.1, 0.4)
    ]
    assert e1.choose(rows, "aucreshape") == {"boost": 0.4}


def _tabular(seed: int = 0) -> replication.Data:
    gen = torch.Generator().manual_seed(seed)

    def part(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = (torch.rand(n, generator=gen) < 0.2).float()
        return torch.randn(n, 4, generator=gen) + y[:, None], y

    (xt, yt), (xv, yv), (xs, ys) = part(1200), part(300), part(300)
    return replication.Data(
        {"train": xt, "val": xv, "test": xs}, {"train": yt, "val": yv, "test": ys}, {"fraud": 0},
        image=False,
    )  # fmt: skip


def _job(epochs: int = 6) -> replication.Job:
    return replication.Job(
        name="t", size=0, arch="mlp", epochs=epochs, lr=0.01, batch_size=32, optimizer="sgd",
        pos_weight=torch.tensor(5.0), window_size=1024, eval_every=1,
    )  # fmt: skip


def _without_test_data(data: replication.Data) -> replication.Data:
    """The same data, with the test cases and labels replaced by noise."""
    gen = torch.Generator().manual_seed(99)
    x = dict(data.x)
    y = dict(data.y)
    x["test"] = torch.randn_like(data.x["test"])
    y["test"] = data.y["test"][torch.randperm(len(data.y["test"]), generator=gen)]
    return replace(data, x=x, y=y)


def test_real_data_selection_does_not_depend_on_test_data() -> None:
    data = _tabular()
    first = replication.train_one(_job(), data, "ar_paper", 0, "cpu")
    second = replication.train_one(_job(), _without_test_data(data), "ar_paper", 0, "cpu")
    assert first["best"]["val_sens"][1] == second["best"]["val_sens"][1]
    assert first["best"]["val_loss"][1] == second["best"]["val_loss"][1]
    for one, two in zip(first["history"], second["history"], strict=True):
        assert one["val_sens"] == two["val_sens"]
        assert one["val_loss"] == two["val_loss"]
        assert one["train_loss"] == two["train_loss"]
    assert max(h["boosted_fraction"][0] for h in first["history"]) > 0  # boosting really happened


def test_real_data_arms_share_initialization_and_batch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A boost of zero makes AUCReshaping the baseline, so any difference from CE would have to come
    # from the initialization, the batch order or the dropout masks.
    arms = dict(replication.ARMS)
    arms["zero_boost"] = {"target_specificity": 0.9, "boost": 0.0}
    monkeypatch.setattr(replication, "ARMS", arms)
    data = _tabular()
    ce = replication.train_one(_job(), data, "ce", 3, "cpu")
    zero = replication.train_one(_job(), data, "zero_boost", 3, "cpu")
    torch.testing.assert_close(zero["last"], ce["last"])
    assert [h["train_loss"] for h in zero["history"]] == pytest.approx(
        [h["train_loss"] for h in ce["history"]]
    )


def test_synthetic_selection_does_not_depend_on_test_data(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(0)

    def part(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = (torch.rand(n, generator=gen) < 0.3).float()
        return torch.randn(n, 4, generator=gen) + y[:, None], y

    (xt, yt), (xv, yv), (xs, ys) = part(600), part(300), part(300)
    spec = runner.RunSpec(
        task="tiny", arm="aucreshape", params={"boost": 0.3, "min_negatives": 20}, seed=0,
        hidden=(8,), dropout=0.1, batch_size=64, epochs=4, checkpoints=(2, 4), target=0.9,
    )  # fmt: skip
    real = runner.Task("tiny", xt, yt, xv, yv, xs, ys)
    noise = runner.Task("tiny", xt, yt, xv, yv, torch.randn_like(xs), 1 - ys)
    first = runner.run(real, spec, tmp_path / "a")
    second = runner.run(noise, spec, tmp_path / "b")
    assert first is not None
    assert second is not None
    for one, two in zip(first["history"], second["history"], strict=True):
        for field in ("epoch", "train_loss", "val_loss", "val_sens", "boosted_fraction"):
            assert one[field] == two[field]
    assert analysis.selected_epoch(first) == analysis.selected_epoch(second)
    assert max(h["boosted_fraction"] for h in first["history"]) > 0  # boosting really happened
    assert np.isfinite(first["history"][-1]["val_sens"])
