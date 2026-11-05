"""Smoke test for benchmarks.review.diagrams (the README's how-it-works figure)."""

from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import pytest

from benchmarks.review import diagrams, figures

# The figures are typeset by LaTeX (the project's figure style), and CI has no LaTeX installed.
pytestmark = pytest.mark.skipif(
    shutil.which("latex") is None, reason="drawing these figures needs a LaTeX installation"
)


@pytest.mark.parametrize("theme", [figures.LIGHT, figures.DARK], ids=["light", "dark"])
def test_how_it_works_draws_both_themes(theme: figures.Theme) -> None:
    figures.apply(theme)
    fig = diagrams.how_it_works(theme)
    texts = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "95" in texts
    assert "boosted" in " ".join(a.get_text() for a in fig.axes[0].texts)
    plt.close(fig)
