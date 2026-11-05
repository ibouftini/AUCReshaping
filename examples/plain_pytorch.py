"""Plain PyTorch: pretrain with BCE, then fine-tune with AUCReshaping (the paper's recipe)."""

from __future__ import annotations

import torch
from _common import batches, make_data, make_model, parse_args
from torch import nn

from aucreshape import AUCReshapingLoss, metrics


def train(model, criterion, x, y, epochs, lr, seed=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for xb, yb in batches(x, y, 256, seed + epoch):
            loss = criterion(model(xb).squeeze(-1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if isinstance(criterion, AUCReshapingLoss):
            criterion.end_epoch()


def main() -> None:
    args = parse_args(__doc__)
    n = 4_000 if args.fast else 40_000
    x, y = make_data(n)
    x_test, y_test = make_data(n, seed=1)
    torch.manual_seed(0)
    model = make_model()

    train(model, nn.BCEWithLogitsLoss(), x, y, epochs=3 if args.fast else 20, lr=1e-3)
    with torch.no_grad():
        before = metrics.sensitivity_at_specificity(model(x_test).squeeze(-1), y_test, 0.95)

    criterion = AUCReshapingLoss(target_specificity=0.95, boost=0.2)  # the only change
    train(model, criterion, x, y, epochs=2 if args.fast else 5, lr=3e-4, seed=100)
    with torch.no_grad():
        after = metrics.sensitivity_at_specificity(model(x_test).squeeze(-1), y_test, 0.95)
    print(f"sensitivity at 95% specificity: before {before:.3f}, after {after:.3f}")
    print(f"threshold (logit) at 0.95: {criterion.thresholds.item():.3f}")


if __name__ == "__main__":
    main()
