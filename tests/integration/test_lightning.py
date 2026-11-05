"""Lightning callback."""

from __future__ import annotations

import importlib
import sys

import pytest
import torch
from torch import nn

from aucreshape import AUCReshapingLoss

L = pytest.importorskip("lightning")

from aucreshape.integrations.lightning import AUCReshapingCallback  # noqa: E402


class Classifier(L.LightningModule):
    def __init__(self, strategy="window"):
        super().__init__()
        self.net = nn.Linear(4, 1)
        self.criterion = AUCReshapingLoss(
            0.9, boost=0.2, strategy=strategy, min_negatives=5, initial_thresholds=[0.0]
        )

    def training_step(self, batch, _):
        x, y = batch
        return self.criterion(self.net(x).squeeze(-1), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def loader():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 4, generator=g)
    y = (torch.rand(64, generator=g) < 0.3).float()
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=16)


def trainer(**kwargs):
    return L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def test_callback_logs_and_ends_epochs():
    model = Classifier()
    t = trainer(max_epochs=2, callbacks=[AUCReshapingCallback()])
    t.fit(model, loader())
    keys = set(t.callback_metrics)
    for stat in ("boosted_fraction", "ready_fraction", "mean_boost", "threshold"):
        assert f"aucreshape/criterion/{stat}" in keys
    assert model.criterion.step.item() == 8


def test_callback_finalises_epoch_strategy():
    model = Classifier(strategy="epoch")
    t = trainer(max_epochs=1, callbacks=[AUCReshapingCallback(log_stats=False)])
    t.fit(model, loader())
    assert model.criterion.tracker.support.item() > 0
    assert not t.callback_metrics


def test_fast_dev_run():
    trainer(fast_dev_run=True, callbacks=[AUCReshapingCallback()]).fit(Classifier(), loader())


def test_criterion_state_is_in_lightning_checkpoint(tmp_path):
    model = Classifier()
    t = trainer(max_epochs=1)
    t.fit(model, loader())
    path = tmp_path / "model.ckpt"
    t.save_checkpoint(path)
    state = torch.load(path, weights_only=False)["state_dict"]
    assert "criterion.step" in state
    assert "criterion.tracker.buffer" in state


def test_helpful_error_without_lightning(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightning", None)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", None)
    monkeypatch.delitem(sys.modules, "aucreshape.integrations.lightning", raising=False)
    with pytest.raises(ImportError, match=r'pip install "aucreshape\[lightning\]"'):
        importlib.import_module("aucreshape.integrations.lightning")
