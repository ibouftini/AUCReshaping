"""DDP correctness: 2 gloo processes on CPU equal one process on the concatenated data."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aucreshape import AUCReshapingLoss, ThresholdTracker

pytestmark = [
    pytest.mark.distributed,
    # Small shards: calibrate/end_epoch warn about few negatives, which is expected here.
    pytest.mark.filterwarnings("ignore:ThresholdTracker:UserWarning"),
]

WORLD = 2
STEPS = 12
SPECS = [0.9, 0.95]


def shard(rank: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-specific batch; sizes differ per rank to exercise variable-size gathering."""
    g = torch.Generator().manual_seed(1000 * step + rank)
    n = 40 + 13 * rank
    y = (torch.rand(n, 2, generator=g) < 0.3).float()
    z = torch.randn(n, 2, generator=g) + y
    return z, y


def make(strategy: str) -> ThresholdTracker:
    return ThresholdTracker(
        SPECS, num_labels=2, strategy=strategy, min_negatives=20, window_size=300, max_samples=250
    )


def worker(rank: int, port: int, strategy: str, out: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        tracker = make(strategy)
        for step in range(STEPS):
            tracker.update(*shard(rank, step))
        tracker.end_epoch()
        val = shard(rank, 99)
        fixed = ThresholdTracker.from_scores(val[0], val[1], SPECS)
        criterion = AUCReshapingLoss(SPECS, num_labels=2, min_negatives=20, window_size=300)
        for step in range(3):
            z, y = shard(rank, step)
            criterion(z.requires_grad_(), y)
        torch.save(
            {
                "thresholds": tracker.thresholds,
                "fixed": fixed.thresholds,
                "loss_thresholds": criterion.thresholds,
            },
            f"{out}/rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.parametrize("strategy", ["window", "epoch", "ema", "batch"])
def test_ddp_thresholds_match_single_process(tmp_path, strategy):
    mp.spawn(worker, args=(free_port(), strategy, str(tmp_path)), nprocs=WORLD, join=True)
    results = [torch.load(tmp_path / f"rank{r}.pt") for r in range(WORLD)]

    single = make(strategy)
    for step in range(STEPS):
        parts = [shard(r, step) for r in range(WORLD)]
        single.update(torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))
    single.end_epoch()
    val = [shard(r, 99) for r in range(WORLD)]
    fixed = ThresholdTracker.from_scores(
        torch.cat([v[0] for v in val]), torch.cat([v[1] for v in val]), SPECS
    )
    criterion = AUCReshapingLoss(SPECS, num_labels=2, min_negatives=20, window_size=300)
    for step in range(3):
        parts = [shard(r, step) for r in range(WORLD)]
        z = torch.cat([p[0] for p in parts]).requires_grad_()
        criterion(z, torch.cat([p[1] for p in parts]))

    for result in results:  # identical on every rank, and equal to the single process
        torch.testing.assert_close(result["thresholds"], single.thresholds)
        torch.testing.assert_close(result["fixed"], fixed.thresholds)
        torch.testing.assert_close(result["loss_thresholds"], criterion.thresholds)
