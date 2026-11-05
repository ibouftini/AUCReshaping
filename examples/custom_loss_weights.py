"""Use only the reshaping weights with your own per-sample loss (here: focal loss)."""

from __future__ import annotations

import torch
from _common import batches, make_data, make_model, parse_args

from aucreshape import ThresholdTracker, metrics, reshaping_weights


def focal_loss(logits, targets, gamma=2.0):
    """Per-sample sigmoid focal loss (Lin et al., 2017), reduction='none'."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    return (1 - p_t) ** gamma * bce


def main() -> None:
    args = parse_args(__doc__)
    x, y = make_data(3_000 if args.fast else 30_000)
    torch.manual_seed(0)
    model = make_model()
    tracker = ThresholdTracker(0.95)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(2 if args.fast else 10):
        model.train()
        for xb, yb in batches(x, y, 256, epoch):
            logits = model(xb).squeeze(-1)
            w = reshaping_weights(logits, yb, tracker.active_thresholds(), boost=1.0)
            loss = (w * focal_loss(logits, yb)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tracker.update(logits, yb)  # after using the thresholds, as the loss module does
    with torch.no_grad():
        sens = metrics.sensitivity_at_specificity(model(x).squeeze(-1), y, 0.95)
    print(
        f"ready={tracker.ready.item()}, threshold={tracker.thresholds.item():.3f}, "
        f"train sensitivity at 0.95 specificity={sens:.3f}"
    )


if __name__ == "__main__":
    main()
