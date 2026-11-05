"""GPU-only checks. Skipped automatically without CUDA."""

from __future__ import annotations

import contextlib

import pytest
import torch

from aucreshape import AUCReshapingLoss, ThresholdTracker, auc_reshaping_loss

pytestmark = pytest.mark.gpu


@contextlib.contextmanager
def no_host_sync():
    """Raise on any synchronising CUDA call (inputs must be on the GPU beforehand)."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def cuda_batch(c=3, n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, c, generator=g).cuda().requires_grad_()
    y = (torch.rand(n, c, generator=g) < 0.3).float().cuda()
    mask = (torch.rand(n, c, generator=g) < 0.9).cuda()
    return z, y, mask


@pytest.mark.parametrize("mode", ["prob", "logit", "weight"])
def test_functional_forward_backward_has_no_host_sync(mode):
    z, y, mask = cuda_batch()
    theta = torch.tensor([[-0.5, 0.5]] * 3, device="cuda")
    boost = torch.tensor([0.3, 0.1], device="cuda")
    with no_host_sync():
        loss = auc_reshaping_loss(z, y, theta, boost, mode=mode, mask=mask, pos_weight=2.0)
        loss.backward()


@pytest.mark.parametrize("strategy", ["fixed", "batch", "window", "ema", "epoch"])
def test_module_training_step_has_no_host_sync(strategy):
    kwargs = {"initial_thresholds": [0.0, 0.5]} if strategy == "fixed" else {"min_negatives": 20}
    criterion = AUCReshapingLoss(
        [0.9, 0.95], boost=[0.3, 0.1], num_labels=3, strategy=strategy, **kwargs
    ).cuda()
    z, y, mask = cuda_batch()
    criterion(z, y, mask).backward()  # first call outside the check (lazy CUDA init)
    batches = [cuda_batch(seed=seed) for seed in range(1, 4)]
    with no_host_sync():
        for z, y, mask in batches:
            criterion(z, y, mask).backward()
            _ = criterion.last_stats
    assert bool(criterion.tracker.ready.all()) or strategy == "epoch"


def test_tracker_update_has_no_host_sync():
    tracker = ThresholdTracker(0.95, num_labels=3, min_negatives=20).cuda()
    batches = [cuda_batch(seed=seed) for seed in range(3)]
    with no_host_sync():
        for z, y, mask in batches:
            tracker.update(z, y, mask)


def test_cuda_matches_cpu_module():
    cpu = AUCReshapingLoss(0.9, boost=0.2, num_labels=3, min_negatives=20)
    gpu = AUCReshapingLoss(0.9, boost=0.2, num_labels=3, min_negatives=20).cuda()
    for seed in range(4):
        z, y, mask = cuda_batch(seed=seed)
        a = cpu(z.detach().cpu().requires_grad_(), y.cpu(), mask.cpu())
        b = gpu(z, y, mask)
        torch.testing.assert_close(b.cpu(), a, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gpu.thresholds.cpu(), cpu.thresholds, rtol=1e-5, atol=1e-6)
