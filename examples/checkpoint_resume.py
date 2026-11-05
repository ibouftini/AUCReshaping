"""Resuming from a checkpoint gives exactly the same training as running without interruption."""

from __future__ import annotations

import io

import torch
from _common import batches, make_data, make_model, parse_args

from aucreshape import AUCReshapingLoss


def run(x, y, steps, state=None):
    torch.manual_seed(0)
    model = make_model()
    criterion = AUCReshapingLoss(0.95, boost=0.2, window_size=1000)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    if state is not None:
        model.load_state_dict(state["model"])
        criterion.load_state_dict(state["criterion"])
        optimizer.load_state_dict(state["optimizer"])
    data = list(batches(x, y, 128, seed=0))
    start = 0 if state is None else state["step"]
    for xb, yb in data[start:steps]:
        loss = criterion(model(xb).squeeze(-1), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, criterion, optimizer


def main() -> None:
    args = parse_args(__doc__)
    x, y = make_data(3_000 if args.fast else 20_000)
    total = len(x) // 128
    half = total // 2
    ref_model, ref_criterion, _ = run(x, y, total)

    model, criterion, optimizer = run(x, y, half)
    buffer = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": half,
        },
        buffer,
    )
    buffer.seek(0)
    resumed_model, resumed_criterion, _ = run(x, y, total, state=torch.load(buffer))

    torch.testing.assert_close(resumed_criterion.thresholds, ref_criterion.thresholds)
    for a, b in zip(resumed_model.parameters(), ref_model.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    print(
        f"resumed run identical: threshold {resumed_criterion.thresholds.item():.4f}, "
        f"step {resumed_criterion.step.item()}"
    )


if __name__ == "__main__":
    main()
