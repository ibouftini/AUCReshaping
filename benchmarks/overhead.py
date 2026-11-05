"""G17 microbenchmark: forward + backward time of a loss step relative to BCEWithLogitsLoss.

Run: uv run python -m benchmarks.overhead [cpu|cuda] [--compile]
"""

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable

import torch
from torch import Tensor, nn

from aucreshape import AUCReshapingLoss, auc_reshaping_loss

LossFn = Callable[[Tensor, Tensor], Tensor]


def timed(loss_fn: LossFn, z: Tensor, y: Tensor, device: str, repeats: int) -> float:
    """Mean seconds per forward + backward step."""

    def step() -> None:
        z.grad = None
        loss_fn(z, y).backward()

    for _ in range(5):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def candidates(c: int, device: str, compile_fn: bool) -> dict[str, LossFn]:
    theta = torch.tensor([[-0.5, 0.5]] * c, device=device)
    boost = torch.tensor([0.2, 0.1], device=device)

    def functional(z: Tensor, y: Tensor) -> Tensor:
        return auc_reshaping_loss(z, y, theta, boost)

    out: dict[str, LossFn] = {"functional": functional}
    if compile_fn:
        out["functional (compiled)"] = torch.compile(functional, fullgraph=True)
    for strategy in ("fixed", "window", "ema", "epoch"):
        kwargs = {"initial_thresholds": [-0.5, 0.5]} if strategy == "fixed" else {}
        out[f"module {strategy}"] = AUCReshapingLoss(
            [0.9, 0.95], boost=[0.2, 0.1], num_labels=c, strategy=strategy, **kwargs
        ).to(device)
    return out


def main(device: str, compile_fn: bool) -> None:
    warnings.simplefilter("ignore", UserWarning)
    torch.manual_seed(0)
    where = (
        torch.cuda.get_device_name(0) if device == "cuda" else f"{torch.get_num_threads()} threads"
    )
    print(f"device {device} ({where}), torch {torch.__version__}")
    names = list(candidates(1, device, compile_fn))
    print("| N | C | BCE (us) | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 3) + "|")
    for n in (256, 4096, 65536):
        for c in (1, 14):
            z = torch.randn(n, c, device=device, requires_grad=True)
            y = (torch.rand(n, c, device=device) < 0.2).float()
            repeats = 200 if n < 65536 else 50
            base = timed(nn.BCEWithLogitsLoss(), z, y, device, repeats)
            cells = []
            for fn in candidates(c, device, compile_fn).values():
                seconds = timed(fn, z, y, device, repeats)
                cells.append(f"{seconds / base:.2f}x ({seconds * 1e6:.0f} us)")
            print(f"| {n} | {c} | {base * 1e6:.0f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(next((a for a in args if a in ("cpu", "cuda")), "cpu"), "--compile" in args)
