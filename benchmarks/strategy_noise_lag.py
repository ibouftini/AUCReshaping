"""Threshold noise vs lag of each tracker strategy on drifting synthetic scores.

Negatives' logits are N(mu_t, 1) with mu_t drifting linearly from 0 to 2 over training (as
scores move during fine-tuning); the true threshold at specificity s is mu_t + Phi^-1(s).
For each strategy we report, after the tracker is ready: mean error (lag/bias), the standard
deviation of the error (noise) and the RMSE. Run: uv run python -m benchmarks.strategy_noise_lag
"""

from __future__ import annotations

import math

import torch

from aucreshape import ThresholdTracker

SPEC = 0.95
STEPS = 2000
BATCH = 64
NEG_RATE = 0.8
SEEDS = 5


def run(strategy: str, seed: int, **kwargs: float) -> tuple[float, float, float, float]:
    g = torch.Generator().manual_seed(seed)
    tracker = ThresholdTracker(SPEC, strategy=strategy, **kwargs)  # type: ignore[arg-type]
    z_s = math.sqrt(2) * torch.erfinv(torch.tensor(2 * SPEC - 1)).item()
    errors = []
    steps_per_epoch = 200
    for t in range(STEPS):
        mu = 2.0 * t / STEPS
        y = (torch.rand(BATCH, generator=g) > NEG_RATE).float()
        z = torch.randn(BATCH, generator=g) + mu + 2.0 * y
        if tracker.ready.item():
            errors.append(tracker.thresholds.item() - (mu + z_s))
        tracker.update(z, y)
        if strategy == "epoch" and (t + 1) % steps_per_epoch == 0:
            tracker.end_epoch()
    e = torch.tensor(errors)
    first_ready = STEPS - len(errors)
    return e.mean().item(), e.std().item(), e.pow(2).mean().sqrt().item(), first_ready


CONFIGS = {
    "batch (min_negatives=40)": ("batch", {"min_negatives": 40}),
    "ema (m=0.9)": ("ema", {"momentum": 0.9}),
    "ema (m=0.99)": ("ema", {"momentum": 0.99}),
    "window (4096, default)": ("window", {}),
    "window (1024)": ("window", {"window_size": 1024, "min_negatives": 200}),
    "epoch (200 steps/epoch)": ("epoch", {}),
}

if __name__ == "__main__":
    print(f"specificity {SPEC}, batch {BATCH}, {STEPS} steps, drift 0 -> 2 logits, {SEEDS} seeds")
    print("| strategy | bias (lag) | noise (std) | RMSE | first ready step |")
    print("|---|---|---|---|---|")
    for name, (strategy, kwargs) in CONFIGS.items():
        rows = torch.tensor([run(strategy, s, **kwargs) for s in range(SEEDS)])
        bias, noise, rmse, ready = rows.mean(0).tolist()
        print(f"| {name} | {bias:+.3f} | {noise:.3f} | {rmse:.3f} | {ready:.0f} |")
