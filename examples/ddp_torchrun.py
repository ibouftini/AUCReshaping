"""DDP on CPU: run with `torchrun --nproc_per_node=2 examples/ddp_torchrun.py --fast`.

Nothing changes in the training code: the tracker all-gathers negatives' logits, so every rank
holds the same thresholds (checked at the end).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from _common import batches, make_data, make_model, parse_args
from torch.nn.parallel import DistributedDataParallel

from aucreshape import AUCReshapingLoss


def main() -> None:
    args = parse_args(__doc__)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    x, y = make_data(4_000 if args.fast else 40_000)
    x, y = x[rank::world], y[rank::world]  # each rank sees its own shard
    torch.manual_seed(0)
    model = DistributedDataParallel(make_model())
    criterion = AUCReshapingLoss(0.95, boost=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(2 if args.fast else 5):
        for xb, yb in batches(x, y, 128, seed=epoch):
            loss = criterion(model(xb).squeeze(-1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    gathered = [torch.zeros_like(criterion.thresholds) for _ in range(world)]
    dist.all_gather(gathered, criterion.thresholds)
    assert all(torch.equal(g, gathered[0]) for g in gathered), "ranks disagree"
    if rank == 0:
        print(
            f"{world} ranks agree: threshold {criterion.thresholds.item():.4f}, "
            f"step {criterion.step.item()}, pid {os.getpid()}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
