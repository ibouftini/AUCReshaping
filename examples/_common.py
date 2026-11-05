"""Tiny synthetic data and helpers shared by the examples (no downloads)."""

from __future__ import annotations

import argparse

import torch
from torch import nn


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--fast", action="store_true", help="tiny run for CI")
    return parser.parse_args()


def make_data(n: int, num_labels: int = 1, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Imbalanced data: 10 features, ~10% positives per label, partly hard to separate."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 10, generator=g)
    weights = torch.randn(10, num_labels, generator=g)
    score = x @ weights + 0.8 * torch.sin(3 * x[:, :num_labels])
    y = (score > torch.quantile(score, 0.9, dim=0)).float()
    return x, (y.squeeze(-1) if num_labels == 1 else y)


def make_model(num_labels: int = 1) -> nn.Module:
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, num_labels))


def batches(x: torch.Tensor, y: torch.Tensor, size: int, seed: int):
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    for i in range(0, len(x), size):
        idx = order[i : i + size]
        yield x[idx], y[idx]
