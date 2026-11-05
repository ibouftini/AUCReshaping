"""Explanatory diagrams for the README and the documentation.

    uv run --group bench python -m benchmarks.review.diagrams

Writes ``benchmarks/results/figures/diagrams/how-it-works{.svg,.pdf,-dark.svg}``; the SVGs are
copied
to ``docs/figures/`` for the README and the docs site.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.review import figures
from benchmarks.review.figures import Theme

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "figures" / "diagrams"
DOCS = ROOT / "docs" / "figures"


def _density(z: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((z - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def how_it_works(theme: Theme) -> plt.Figure:
    """Negatives' and positives' logits, the 95%-specificity threshold and the boosted set."""
    z = np.linspace(-4.8, 5.0, 1000)
    neg_mean, neg_sd = -1.5, 1.0
    neg = _density(z, neg_mean, neg_sd)
    pos = 0.8 * _density(z, 1.2, 1.2)
    theta = neg_mean + 1.6449 * neg_sd  # 95% of the negatives lie below
    purple, grey = theme.roles["aucreshape"], theme.roles["ce"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.fill_between(z, neg, color=grey, alpha=0.22, lw=0)
    ax.plot(z, neg, color=grey, lw=1.2)
    below = z <= theta
    ax.fill_between(z[below], pos[below], color=purple, alpha=0.85, lw=0)
    ax.fill_between(z[~below], pos[~below], color=purple, alpha=0.22, lw=0)
    ax.plot(z, pos, color=purple, lw=1.2)
    top = 1.18 * neg.max()
    ax.plot([theta, theta], [0, top], color=theme.ink, lw=0.9, ls=(0, (3, 2)))
    ax.text(theta + 0.08, top, r"$\theta$: 95\% of the negatives lie below",
            va="top", ha="left", fontsize=8)  # fmt: skip
    ax.text(neg_mean, 1.03 * neg.max(), "negatives (label 0)", ha="center", va="bottom",
            fontsize=8, color=theme.muted)  # fmt: skip
    ax.text(2.6, 0.8 * _density(np.array(2.6), 1.2, 1.2) + 0.03, "positives (label 1)",
            ha="left", va="bottom", fontsize=8, color=theme.muted)  # fmt: skip
    ax.text(1.6, 0.06, "caught: plain BCE", ha="center", va="bottom", fontsize=7.5, color=theme.ink)
    tail = theta - 0.45
    ax.annotate(
        "missed at 95\\% specificity:\ntheir loss is boosted",
        xy=(tail, 0.45 * 0.8 * _density(np.array(tail), 1.2, 1.2)),
        xytext=(-4.6, 0.30),
        fontsize=8,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-|>", "color": theme.ink, "lw": 0.8, "shrinkB": 2},
    )
    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(0, 1.22 * neg.max())
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("logit (model score)")
    fig.tight_layout()
    return fig


def main() -> None:
    import shutil

    for path in figures.render(how_it_works, "how-it-works", OUT):
        print(path)
        if path.suffix == ".svg":
            shutil.copy2(path, DOCS / path.name)


if __name__ == "__main__":
    main()
