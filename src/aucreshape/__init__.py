"""AUCReshaping for PyTorch: higher sensitivity at high specificity.

Boost the loss of positive samples that are misclassified at a high-specificity threshold
(Bhat et al., *Scientific Reports* 13, 21097, 2023, doi:10.1038/s41598-023-48482-x).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from aucreshape import functional, metrics
from aucreshape.functional import (
    BoostSchedule,
    auc_reshaping_loss,
    boost_values,
    linear_ramp,
    reshaping_weights,
)
from aucreshape.loss import AUCReshapingLoss
from aucreshape.thresholds import ThresholdTracker

try:
    __version__ = version("aucreshape")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "AUCReshapingLoss",
    "BoostSchedule",
    "ThresholdTracker",
    "__version__",
    "auc_reshaping_loss",
    "boost_values",
    "functional",
    "linear_ramp",
    "metrics",
    "reshaping_weights",
]
