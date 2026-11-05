"""Input checks shared by the public functions.

Only metadata (shapes, dtypes, Python values) is inspected here, never tensor values, so
nothing in this module forces a host synchronisation.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch
from torch import Tensor


def check_choice(name: str, value: str, options: Sequence[str]) -> None:
    """Raise a ValueError if ``value`` is not one of ``options``."""
    if value not in options:
        valid = ", ".join(repr(o) for o in options)
        raise ValueError(f"{name} must be one of {valid}; got {value!r}.")


def as_label_matrix(
    logits: Tensor, targets: Tensor, mask: Tensor | None
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Validate shapes and return 2-D ``(N, C)`` views of logits, targets and mask.

    ``(N,)`` inputs become ``(N, 1)``. A ``(N,)`` versus ``(N, 1)`` mismatch between
    ``logits`` and ``targets``/``mask`` is accepted and follows ``logits``.
    """
    if not isinstance(logits, Tensor):
        raise TypeError(f"logits must be a torch.Tensor; got {type(logits).__name__}.")
    if not logits.is_floating_point():
        raise TypeError(
            f"logits must be a floating-point tensor (raw model outputs); got {logits.dtype}. "
            "Convert with logits.float()."
        )
    if logits.dim() not in (1, 2):
        raise ValueError(
            "logits must have 1 or 2 dimensions, (N,) for binary or (N, C) for multi-label; "
            f"got shape {tuple(logits.shape)}."
        )
    z = logits if logits.dim() == 2 else logits.unsqueeze(-1)
    y = _match_shape("targets", targets, logits, z)
    m = None if mask is None else _match_shape("mask", mask, logits, z)
    return z, y, m


def _match_shape(name: str, other: Tensor, logits: Tensor, z: Tensor) -> Tensor:
    if not isinstance(other, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(other).__name__}.")
    if other.shape == logits.shape or other.shape == z.shape:
        return other.reshape(z.shape)
    if logits.dim() == 2 and logits.shape[1] == 1 and other.shape == logits.shape[:1]:
        return other.reshape(z.shape)
    raise ValueError(
        f"{name} must have the same shape as logits {tuple(logits.shape)}; "
        f"got {tuple(other.shape)}."
    )


def check_specificities(target_specificity: float | Sequence[float]) -> tuple[float, ...]:
    """Validate target specificities: in (0, 1), strictly ascending, at least one."""
    if isinstance(target_specificity, (int, float)):
        values: tuple[float, ...] = (float(target_specificity),)
    else:
        values = tuple(float(s) for s in target_specificity)
    if not values:
        raise ValueError("target_specificity needs at least one value in (0, 1).")
    for s in values:
        if not 0.0 < s < 1.0:
            raise ValueError(
                f"target_specificity values must be in (0, 1); got {s}.{percent_hint(s)}"
            )
    if any(b <= a for a, b in itertools.pairwise(values)):
        raise ValueError(
            "target_specificity must be strictly ascending (lowest to highest specificity); "
            f"got {list(values)}."
        )
    return values


def percent_hint(value: float) -> str:
    """Suggest a fraction when a specificity looks like a percentage."""
    return f" Did you mean {value / 100:g}?" if 1.0 < value <= 100.0 else ""


def check_nonnegative_floats(name: str, values: Sequence[float]) -> None:
    """Raise if any Python number in ``values`` is negative or not finite."""
    for v in values:
        if not (math.isfinite(v) and v >= 0.0):
            raise ValueError(f"{name} must be finite and non-negative; got {list(values)}.")


def check_positive_int(name: str, value: int) -> None:
    """Raise if ``value`` is not a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")


def compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the internal precision: float64 stays float64, all else becomes float32."""
    return torch.float64 if dtype == torch.float64 else torch.float32
