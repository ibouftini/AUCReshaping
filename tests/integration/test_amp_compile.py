"""AMP and torch.compile."""

from __future__ import annotations

import shutil

import pytest
import torch
from torch import nn

from aucreshape import AUCReshapingLoss, auc_reshaping_loss

# Inductor needs a C++ compiler on CPU; without one, aot_eager still checks graph capture
# (fullgraph=True means no graph breaks) and AOT autograd tracing.
BACKEND = "inductor" if shutil.which("g++") or shutil.which("c++") else "aot_eager"


def data(n=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    y = (torch.rand(n, generator=g) < 0.4).float()
    return x, y


@pytest.mark.parametrize("mode", ["prob", "logit", "weight"])
def test_cpu_bfloat16_autocast(mode):
    x, y = data()
    model = nn.Linear(8, 1)
    criterion = AUCReshapingLoss(0.9, boost=0.3, mode=mode, initial_thresholds=[0.0])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x).squeeze(-1)
        assert logits.dtype == torch.bfloat16
        loss = criterion(logits, y)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert criterion.tracker.buffer.dtype == torch.float32


@pytest.mark.gpu
def test_cuda_float16_autocast_and_grad_scaler():
    x, y = data()
    model = nn.Linear(8, 1).cuda()
    criterion = AUCReshapingLoss(0.9, boost=0.3, initial_thresholds=[0.0])
    scaler = torch.amp.GradScaler("cuda")
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = criterion(model(x.cuda()).squeeze(-1), y.cuda())
    assert loss.dtype == torch.float32
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert torch.isfinite(loss)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["prob", "logit", "weight"])
@pytest.mark.parametrize("combine", ["band", "sum", "max"])
def test_functional_compiles_fullgraph(mode, combine):
    torch._dynamo.reset()
    z = torch.randn(64, 3, requires_grad=True)
    y = (torch.rand(64, 3) < 0.5).float()
    theta = torch.tensor([[-0.5, 0.5]] * 3)

    def fn(z, y, theta):
        return auc_reshaping_loss(z, y, theta, [0.3, 0.1], mode=mode, combine=combine)

    compiled = torch.compile(fn, fullgraph=True, backend=BACKEND)
    out = compiled(z, y, theta)
    torch.testing.assert_close(out, fn(z, y, theta))
    (grad,) = torch.autograd.grad(out, z)
    (ref,) = torch.autograd.grad(fn(z, y, theta), z)
    torch.testing.assert_close(grad, ref)


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["fixed", "window"])
def test_module_compiles(strategy):
    torch._dynamo.reset()
    kwargs = {"initial_thresholds": [0.0]} if strategy == "fixed" else {"min_negatives": 10}
    eager = AUCReshapingLoss(0.9, boost=0.2, strategy=strategy, **kwargs)
    compiled_criterion = AUCReshapingLoss(0.9, boost=0.2, strategy=strategy, **kwargs)
    compiled = torch.compile(compiled_criterion, backend=BACKEND)
    for seed in range(3):
        x, y = data(seed=seed)
        z = x[:, 0].clone().requires_grad_()
        torch.testing.assert_close(compiled(z, y), eager(z.detach().clone().requires_grad_(), y))
    torch.testing.assert_close(compiled_criterion.thresholds, eager.thresholds)
