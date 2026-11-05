"""Matplotlib figures for the reports (static PNG, light theme).

Colours are the validated slots 1-3 of the reference categorical palette (blue, orange,
aqua; all-pairs CVD-safe). Every figure has a legend and its numbers also appear in a table
in the report, so identity never relies on colour alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _figure(width: float = 5.6, height: float = 3.6) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def roc_zoom(
    y: np.ndarray,
    curves: dict[str, np.ndarray],
    path: Path,
    title: str,
    max_fpr: float = 0.1,
    marks: Sequence[float] = (0.02, 0.05),
) -> None:
    """ROC curves zoomed to FPR <= max_fpr, with dotted guides at the target FPRs."""
    fig, ax = _figure()
    for i, (name, scores) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, color=SERIES[i], linewidth=2, label=name, drawstyle="steps-post")
    for m in marks:
        ax.axvline(m, color=TEXT_2, linewidth=0.8, linestyle=":")
        ax.text(
            m,
            0.02,
            f" spec {1 - m:g}",
            color=TEXT_2,
            fontsize=7,
            rotation=90,
            va="bottom",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xlim(0, max_fpr)
    _style(ax, title, "False-positive rate", "Sensitivity (TPR)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, path)


def lines(
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    errors: dict[str, Sequence[tuple[float, float]]] | None = None,
    zero_line: bool = False,
) -> None:
    """One or more lines on a single axis, optional (lo, hi) error bars."""
    fig, ax = _figure()
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(x, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=name)
        if errors and name in errors:
            lo = [y - e[0] for y, e in zip(ys, errors[name], strict=True)]
            hi = [e[1] - y for y, e in zip(ys, errors[name], strict=True)]
            ax.errorbar(x, ys, yerr=[lo, hi], color=SERIES[i], linewidth=1, capsize=3, fmt="none")
    if zero_line:
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
    _style(ax, title, xlabel, ylabel)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def forest(
    deltas: dict[str, dict[str, tuple[float, float, float]]],
    labels: dict[str, str],
    path: Path,
    title: str,
) -> None:
    """Small multiples, one per dataset: delta vs baseline with 95% CIs, methods on the y-axis.

    Baselines use slot 1 (blue) and AUCReshaping variants slot 2 (orange); method names are
    on the axis, so colour only groups them.
    """
    datasets = list(deltas)
    methods = [m for m in labels if any(m in d for d in deltas.values())]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(3.0 * len(datasets) + 1.6, 0.42 * len(methods) + 1.4),
        dpi=150,
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        for row, method in enumerate(methods):
            if method not in deltas[dataset]:
                continue
            d, lo, hi = deltas[dataset][method]
            color = SERIES[1] if method.startswith("aucreshape") else SERIES[0]
            ax.plot([lo, hi], [row, row], color=color, linewidth=2)
            ax.plot(
                [d],
                [row],
                "o",
                color=color,
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
        ax.axvline(0, color=TEXT_2, linewidth=0.8)
        _style(ax, dataset, "delta points", "")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([labels[m] for m in methods])
        ax.invert_yaxis()
    handles = [
        plt.Line2D([], [], color=SERIES[0], marker="o", linewidth=2, label="baselines"),
        plt.Line2D([], [], color=SERIES[1], marker="o", linewidth=2, label="AUCReshaping"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.01, ha="left")
    _save(fig, path)
