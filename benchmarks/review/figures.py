"""Figure style of the evaluation.

Colours have fixed meanings across every figure. They come from the maintainer's palette
(purple ``#5A189A``, orange ``#FF9100``, cream ``#FEFAE0``, ink ``#0A0908``), extended with
hues and steps checked by the dataviz validator (CVD separation, normal-vision floor,
lightness band, contrast) on the cream surface (light) and the ink surface (dark):

- AUCReshaping is ``#7B2CBF``, one step lighter than the maintainer's ``#5A189A`` on the same
  ramp: ``#5A189A`` sits just below the light lightness band (OKLCH L 0.397 < 0.43), while
  ``#7B2CBF`` passes every check. ``#5A189A`` remains the dark end of the purple maps;
- orange on cream is 2.15:1, below 3:1 for marks, so orange series always carry a direct
  label and their numbers are in a table next to the figure;
- the dark theme uses the next steps of the same ramps (purple ``#9D4EDD``, orange
  ``#E86A00``); every dark slot passes all checks.

Text and axes wear ink and the neutral, never a series colour. Draw each figure through
:func:`render`, which builds it once per theme and writes ``<name>.svg`` and ``<name>.pdf``
(light) and ``<name>-dark.svg`` (for mkdocs-material's ``#only-dark``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

STYLE = Path(__file__).resolve().parents[1] / "style" / "aucreshape.mplstyle"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    muted: str  # secondary text, axes and the CE baseline
    grid: str
    roles: dict[str, str]
    purple_ramp: tuple[str, ...]  # light -> dark, for AUCReshaping variants and sequential maps
    orange_ramp: tuple[str, ...]
    midpoint: str  # diverging midpoint, a near-neutral of the surface


LIGHT = Theme(
    name="light",
    surface="#FEFAE0",
    ink="#0A0908",
    muted="#6B665C",
    grid="#E9E4CC",
    roles={
        "aucreshape": "#7B2CBF",
        "weighted_bce": "#FF9100",
        "focal": "#2A9D8F",
        "libauc_pauc": "#4361EE",
        "ce": "#6B665C",
    },
    purple_ramp=("#E0AAFF", "#C77DFF", "#9D4EDD", "#7B2CBF", "#5A189A", "#3C096C"),
    orange_ramp=("#FFD8A8", "#FFB55C", "#FF9E00", "#FF9100", "#FF7900", "#E86A00"),
    midpoint="#F2EEE0",
)
DARK = Theme(
    name="dark",
    surface="#0A0908",
    ink="#FEFAE0",
    muted="#A8A395",
    grid="#2A2824",
    roles={
        "aucreshape": "#9D4EDD",
        "weighted_bce": "#E86A00",
        "focal": "#2A9D8F",
        "libauc_pauc": "#4895EF",
        "ce": "#A8A395",
    },
    purple_ramp=("#3C096C", "#5A189A", "#7B2CBF", "#9D4EDD", "#C77DFF", "#E0AAFF"),
    orange_ramp=("#5C2A00", "#8A3F00", "#B85400", "#E86A00", "#FF9100", "#FFB55C"),
    midpoint="#1E1C19",
)
LABELS = {
    "aucreshape": "AUCReshaping",
    "weighted_bce": "weighted BCE",
    "focal": "focal loss",
    "libauc_pauc": "LibAUC pAUC",
    "ce": "CE",
}


def diverging(theme: Theme) -> LinearSegmentedColormap:
    """Orange (AUCReshaping worse) -> neutral -> purple (AUCReshaping better), for delta maps."""
    colours = [theme.orange_ramp[-1], theme.orange_ramp[1], theme.midpoint]
    colours += [theme.purple_ramp[1], theme.purple_ramp[-2]]
    return LinearSegmentedColormap.from_list(f"delta_{theme.name}", colours)


def sequential(theme: Theme) -> LinearSegmentedColormap:
    """Surface -> purple, for magnitudes."""
    return LinearSegmentedColormap.from_list(
        f"magnitude_{theme.name}", [theme.surface, *theme.purple_ramp[:-1]]
    )


def end_labels(
    ax: plt.Axes, x: float, ends: dict[str, float], theme: Theme, min_gap: float = 0.06
) -> None:
    """Direct labels at the right end of lines, pushed apart to at least ``min_gap``
    (a fraction of the y range) so close lines keep readable labels."""
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    placed: list[float] = []
    for text, y in sorted(ends.items(), key=lambda item: item[1]):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.annotate(
            text,
            (x, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=theme.ink,
            annotation_clip=False,
        )


def apply(theme: Theme) -> None:
    """Load the style file, then the theme's colours."""
    plt.style.use(STYLE)
    plt.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink,
            "axes.edgecolor": theme.muted,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "grid.color": theme.grid,
        }
    )


def render(draw: Callable[[Theme], plt.Figure], name: str, out: Path) -> list[Path]:
    """Draw the figure in both themes and save it; return the written paths."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in (LIGHT, DARK):
        apply(theme)
        fig = draw(theme)
        if theme is LIGHT:
            paths += [out / f"{name}.svg", out / f"{name}.pdf"]
            fig.savefig(paths[-2])
            fig.savefig(paths[-1])
        else:
            paths.append(out / f"{name}-dark.svg")
            fig.savefig(paths[-1])
        plt.close(fig)
    return paths
