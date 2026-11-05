"""PyTorch Lightning: the criterion is a submodule (so it is checkpointed) plus the callback."""

from __future__ import annotations

import lightning as L
import torch
from _common import make_data, make_model, parse_args
from torch.utils.data import DataLoader, TensorDataset

from aucreshape import AUCReshapingLoss, metrics
from aucreshape.integrations.lightning import AUCReshapingCallback


class Classifier(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = make_model()
        self.criterion = AUCReshapingLoss(0.95, boost=0.2)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.criterion(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", self.criterion(self(x), y))  # plain BCE outside training

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    args = parse_args(__doc__)
    n = 4_000 if args.fast else 40_000
    x, y = make_data(n)
    x_val, y_val = make_data(n // 2, seed=1)
    torch.manual_seed(0)
    model = Classifier()
    trainer = L.Trainer(
        max_epochs=2 if args.fast else 10,
        accelerator="cpu",
        callbacks=[AUCReshapingCallback()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True),
        DataLoader(TensorDataset(x_val, y_val), batch_size=1024),
    )
    model.eval()
    with torch.no_grad():
        sens = metrics.sensitivity_at_specificity(model(x_val), y_val, 0.95)
    logged = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()}
    print("logged:", logged)
    print(f"val sensitivity at 0.95 specificity: {sens:.3f}")


if __name__ == "__main__":
    main()
