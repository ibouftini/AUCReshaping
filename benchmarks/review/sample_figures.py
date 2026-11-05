"""Sample figures for approving the figure style: the palette sheet and one real result.

    uv run --group bench python -m benchmarks.review.sample_figures [OUT_DIR]

The real result is the first credit-card run (200 epochs, 5 seeds, CE vs AUCReshaping 0.2 at
0.90-0.99 with the default ``band`` rule; paper convention: normal transactions are label 1
and specificity is measured on frauds).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.review import figures
from benchmarks.review.figures import LABELS, Theme

ROOT = Path(__file__).resolve().parents[2]
S11 = ROOT / "benchmarks" / "results" / "replication_local" / "creditcard.jsonl"


def palette_sheet(theme: Theme) -> plt.Figure:
    fig, (top, mid, bottom) = plt.subplots(
        3, 1, figsize=(6.0, 2.6), gridspec_kw={"height_ratios": [1.4, 0.5, 0.5]}
    )
    for i, (role, colour) in enumerate(theme.roles.items()):
        top.add_patch(plt.Rectangle((i, 0.35), 0.8, 0.6, color=colour, lw=0))
        top.text(i, 0.18, LABELS[role], va="top", fontsize=8)
        hex_code = colour.replace("#", r"\#")
        top.text(i, -0.12, rf"\texttt{{{hex_code}}}", va="top", fontsize=7, color=theme.muted)
    top.set_xlim(0, 5)
    top.set_ylim(-0.4, 1.0)
    top.set_title(f"Series roles ({theme.name} theme)")
    gradient = np.linspace(0, 1, 256)[None, :]
    for ax, cmap, label in (
        (mid, figures.sequential(theme), r"Magnitude: surface $\rightarrow$ purple"),
        (
            bottom,
            figures.diverging(theme),
            r"$\Delta$: AUCReshaping worse $\leftarrow 0 \rightarrow$ better",
        ),
    ):
        ax.imshow(gradient, aspect="auto", cmap=cmap, extent=(0, 5, 0, 1))
        ax.set_title(label, fontsize=8, pad=3)
    for ax in (top, mid, bottom):
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def creditcard_epochs(theme: Theme) -> plt.Figure:
    rows = [json.loads(line) for line in S11.read_text().splitlines()]
    metric = "test/fraud/sens@0.98"
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.8)
    ends = {}
    for arm, role in (("ce", "ce"), ("ar_paper_cc", "aucreshape")):
        runs = [r for r in rows if r["arm"] == arm]
        epochs = np.array([h["epoch"] for h in runs[0]["history"]])
        values = np.array([[h[metric] for h in r["history"]] for r in runs])
        colour = theme.roles[role]
        ax.fill_between(epochs, values.min(0), values.max(0), color=colour, alpha=0.10, lw=0)
        ax.plot(epochs, values.mean(0), color=colour)
        ends[LABELS[role]] = float(values.mean(0)[-1])
        chosen = np.median([r["selected"]["val_loss"]["epoch"] for r in runs])
        ax.axvline(chosen, color=colour, lw=0.8, ls=(0, (2, 2)))
    ax.text(
        30,
        0.02,
        "median epoch with the lowest\nvalidation loss (dashed)",
        fontsize=7,
        color=theme.muted,
        va="bottom",
    )
    figures.end_labels(ax, 200, ends, theme)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"test sensitivity at $\mathrm{spec} = 0.98$")
    ax.set_title(r"Credit card, 200 epochs: mean of 5 seeds, band = min--max")
    fig.tight_layout()
    return fig


def main() -> None:
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "benchmarks" / "results" / "figures" / "s12a"
    )
    for name, draw in (("palette", palette_sheet), ("creditcard_epochs", creditcard_epochs)):
        for path in figures.render(draw, name, out):
            print(path)


if __name__ == "__main__":
    main()
