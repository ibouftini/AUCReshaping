"""``torch.distributed`` helpers that are no-ops when not initialised."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


def is_distributed() -> bool:
    """Return True when a process group with more than one rank is initialised."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def all_gather_rows(x: Tensor) -> Tensor:
    """Concatenate ``x`` (shape ``(N_r, ...)``, N_r may differ per rank) over all ranks.

    Rows are ordered by rank, so the result is identical on every rank and equals the
    concatenation a single process would see. Two collectives are issued (sizes, then
    values padded to the largest size); reading the sizes is a host sync. Returns ``x``
    unchanged when not distributed.

    Parameters
    ----------
    x : Tensor, shape (N_r, ...)
        Local rows, on a device supported by the process group's backend.

    Returns
    -------
    Tensor, shape (sum_r N_r, ...)
        Rows of all ranks.
    """
    if not is_distributed():
        return x
    world = dist.get_world_size()
    size = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int64)
    sizes = [torch.zeros_like(size) for _ in range(world)]
    dist.all_gather(sizes, size)
    counts = [int(s.item()) for s in sizes]
    longest = max(counts)
    padded = x.new_zeros((longest, *x.shape[1:]))
    padded[: x.shape[0]] = x
    gathered = [torch.zeros_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    return torch.cat([g[:n] for g, n in zip(gathered, counts, strict=True)], dim=0)
