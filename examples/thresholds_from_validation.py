"""The paper's protocol: thresholds from a validation pass, refreshed every epoch."""

from __future__ import annotations

import torch
from _common import batches, make_data, make_model, parse_args
from torch import nn

from aucreshape import AUCReshapingLoss, ThresholdTracker, metrics


def main() -> None:
    args = parse_args(__doc__)
    n = 4_000 if args.fast else 40_000
    x, y = make_data(n)
    x_val, y_val = make_data(n // 2, seed=1)
    torch.manual_seed(0)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(3 if args.fast else 10):  # pretraining with BCE
        for xb, yb in batches(x, y, 256, epoch):
            loss = bce(model(xb).squeeze(-1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        tracker = ThresholdTracker.from_scores(model(x_val).squeeze(-1), y_val, 0.95)
    criterion = AUCReshapingLoss(boost=0.2, tracker=tracker)
    for epoch in range(2 if args.fast else 5):
        model.train()
        for xb, yb in batches(x, y, 256, 100 + epoch):
            loss = criterion(model(xb).squeeze(-1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val).squeeze(-1)
        criterion.tracker.calibrate(val_logits, y_val)  # refresh the operating point
        sens = metrics.sensitivity_at_specificity(val_logits, y_val, 0.95)
        print(
            f"epoch {epoch}: threshold {criterion.thresholds.item():.3f}, val sens@0.95 {sens:.3f}"
        )


if __name__ == "__main__":
    main()
