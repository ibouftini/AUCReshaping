"""Contract matrix: every combination of device, dtype, shape, mode, strategy and mask."""

from __future__ import annotations

import itertools

import pytest
import torch

from aucreshape import AUCReshapingLoss

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
if torch.backends.mps.is_available():
    DEVICES.append("mps")
DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
SHAPES = [(32,), (32, 1), (32, 3)]
MODES = ["prob", "logit", "weight"]
STRATEGIES = ["fixed", "batch", "window", "ema", "epoch"]
REDUCTIONS = ["mean", "sum", "none"]
MASKS = [False, True]

CASES = [
    case
    for case in itertools.product(DEVICES, DTYPES, SHAPES, MODES, STRATEGIES, REDUCTIONS, MASKS)
    if not (case[0] == "mps" and case[1] == torch.float64)  # MPS has no float64
]


def make_batch(shape, dtype, device, seed):
    g = torch.Generator().manual_seed(seed)
    z = (torch.randn(shape, generator=g) * 3).to(device=device, dtype=dtype)
    y = (torch.rand(shape, generator=g) < 0.3).float().to(device)
    mask = (torch.rand(shape, generator=g) < 0.8).to(device)
    return z, y, mask


@pytest.mark.parametrize(
    ("device", "dtype", "shape", "mode", "strategy", "reduction", "use_mask"),
    CASES,
    ids=lambda v: str(v).replace("torch.", ""),
)
def test_contract(device, dtype, shape, mode, strategy, reduction, use_mask):
    num_labels = 1 if len(shape) == 1 else shape[1]
    criterion = AUCReshapingLoss(
        [0.8, 0.9],
        boost=[0.3, 0.1],
        mode=mode,
        strategy=strategy,
        num_labels=num_labels,
        reduction=reduction,
        min_negatives=5,
        initial_thresholds=[-0.5, 0.5] if strategy == "fixed" else None,
    )
    for step in range(3):  # a few training steps so the trackers become ready
        z, y, mask = make_batch(shape, dtype, device, step)
        z.requires_grad_(True)
        loss = criterion(z, y, mask if use_mask else None)
        expected_shape = shape if reduction == "none" else ()
        assert loss.shape == expected_shape
        assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
        assert loss.device.type == device
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        assert z.grad is not None
        assert z.grad.dtype == dtype
        assert torch.isfinite(z.grad).all()
        criterion.end_epoch()
    assert bool(criterion.tracker.ready.all())

    state = {k: v.clone() for k, v in criterion.state_dict().items() if isinstance(v, torch.Tensor)}
    criterion.eval()
    z, y, mask = make_batch(shape, dtype, device, 99)
    criterion(z, y, mask if use_mask else None)
    with torch.no_grad():
        criterion.train()(z, y, mask if use_mask else None)
    for key, value in criterion.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, state[key])
