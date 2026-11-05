"""Multi-label (5 labels) with missing labels: NaN targets are masked out of loss and tracker."""

from __future__ import annotations

import torch
from _common import batches, make_data, make_model, parse_args

from aucreshape import AUCReshapingLoss, metrics


def main() -> None:
    args = parse_args(__doc__)
    num_labels = 5
    x, y = make_data(4_000 if args.fast else 40_000, num_labels=num_labels)
    missing = torch.rand(y.shape, generator=torch.Generator().manual_seed(3)) < 0.3
    y_raw = y.masked_fill(missing, float("nan"))  # 30% of labels unknown
    torch.manual_seed(0)
    model = make_model(num_labels)
    criterion = AUCReshapingLoss([0.90, 0.95], boost=[0.2, 0.1], num_labels=num_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(2 if args.fast else 10):
        model.train()
        for xb, yb in batches(x, y_raw, 256, epoch):
            mask = ~torch.isnan(yb)
            loss = criterion(model(xb), yb.nan_to_num(0.0), mask=mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(x)
    sens = metrics.sensitivity_at_specificity(logits, y_raw.nan_to_num(0.0), 0.95, mask=~missing)
    print("thresholds (C, K):", criterion.thresholds.numpy().round(2).tolist())
    print("sensitivity at 0.95 per label:", sens.numpy().round(3).tolist())


if __name__ == "__main__":
    main()
