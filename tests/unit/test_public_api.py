"""Pin the public API of version 0.1: exported names and signatures."""

from __future__ import annotations

import inspect

import pytest

import aucreshape
from aucreshape import functional, metrics

EXPECTED_ALL = [
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

COMBINE = "combine: 'Combine' = 'band', mask: 'Tensor | None' = None"
EXPECTED_SIGNATURES = {
    functional.boost_values: (
        "(logits: 'Tensor', targets: 'Tensor', thresholds: 'Tensor', boost: 'Boosts', *, "
        f"{COMBINE}) -> 'Tensor'"
    ),
    functional.reshaping_weights: (
        "(logits: 'Tensor', targets: 'Tensor', thresholds: 'Tensor', boost: 'Boosts', *, "
        f"{COMBINE}) -> 'Tensor'"
    ),
    functional.auc_reshaping_loss: (
        "(logits: 'Tensor', targets: 'Tensor', thresholds: 'Tensor', boost: 'Boosts', *, "
        "mode: 'Mode' = 'prob', combine: 'Combine' = 'band', "
        "pos_weight: 'float | Tensor | None' = None, weight: 'Tensor | None' = None, "
        "mask: 'Tensor | None' = None, reduction: 'Reduction' = 'mean', "
        "min_ratio: 'float' = 0.1) -> 'Tensor'"
    ),
    functional.linear_ramp: "(slope: 'float', max_boost: 'float') -> 'BoostSchedule'",
    aucreshape.ThresholdTracker.__init__: (
        "(self, target_specificity: 'float | Sequence[float]', *, num_labels: 'int' = 1, "
        "strategy: 'Strategy' = 'window', window_size: 'int | None' = None, "
        "momentum: 'float' = 0.9, min_negatives: 'int | None' = None, "
        "max_samples: 'int | None' = None, sync_distributed: 'bool' = True, "
        "initial_thresholds: 'Tensor | Sequence[float] | None' = None, seed: 'int' = 0) "
        "-> 'None'"
    ),
    aucreshape.ThresholdTracker.update: (
        "(self, logits: 'Tensor', targets: 'Tensor', mask: 'Tensor | None' = None) -> 'None'"
    ),
    aucreshape.ThresholdTracker.active_thresholds: "(self) -> 'Tensor'",
    aucreshape.ThresholdTracker.calibrate: (
        "(self, logits: 'Tensor', targets: 'Tensor', mask: 'Tensor | None' = None) -> 'None'"
    ),
    aucreshape.ThresholdTracker.end_epoch: "(self) -> 'None'",
    aucreshape.ThresholdTracker.reset: "(self) -> 'None'",
    aucreshape.ThresholdTracker.from_scores: (
        "(logits: 'Tensor', targets: 'Tensor', "
        "target_specificity: 'float | Sequence[float]', *, mask: 'Tensor | None' = None, "
        "sync_distributed: 'bool' = True) -> 'ThresholdTracker'"
    ),
    aucreshape.AUCReshapingLoss.__init__: (
        "(self, target_specificity: 'float | Sequence[float] | None' = None, "
        "boost: 'float | Sequence[float] | Tensor | BoostSchedule' = 0.1, *, "
        "mode: 'Mode' = 'prob', combine: 'Combine' = 'band', num_labels: 'int | None' = None, "
        "pos_weight: 'float | Tensor | None' = None, weight: 'Tensor | None' = None, "
        "reduction: 'Reduction' = 'mean', "
        "strategy: 'Strategy' = 'window', tracker: 'ThresholdTracker | None' = None, "
        "window_size: 'int | None' = None, momentum: 'float' = 0.9, "
        "min_negatives: 'int | None' = None, max_samples: 'int | None' = None, "
        "sync_distributed: 'bool' = True, "
        "initial_thresholds: 'Tensor | Sequence[float] | None' = None, "
        "auto_update: 'bool | None' = None, warmup_steps: 'int' = 0, "
        "min_ratio: 'float' = 0.1) -> 'None'"
    ),
    aucreshape.AUCReshapingLoss.forward: (
        "(self, logits: 'Tensor', targets: 'Tensor', mask: 'Tensor | None' = None) -> 'Tensor'"
    ),
    aucreshape.AUCReshapingLoss.update: (
        "(self, logits: 'Tensor', targets: 'Tensor', mask: 'Tensor | None' = None) -> 'None'"
    ),
    aucreshape.AUCReshapingLoss.end_epoch: "(self) -> 'None'",
    metrics.threshold_at_specificity: (
        "(scores: 'Tensor', targets: 'Tensor', specificity: 'float | Sequence[float]', *, "
        "mask: 'Tensor | None' = None) -> 'Tensor'"
    ),
    metrics.sensitivity_at_threshold: (
        "(scores: 'Tensor', targets: 'Tensor', threshold: 'float | Tensor', *, "
        "mask: 'Tensor | None' = None) -> 'Tensor'"
    ),
    metrics.sensitivity_at_specificity: (
        "(scores: 'Tensor', targets: 'Tensor', specificity: 'float | Sequence[float]', *, "
        "interpolate: 'bool' = True, mask: 'Tensor | None' = None) -> 'Tensor'"
    ),
    metrics.partial_auc: (
        "(scores: 'Tensor', targets: 'Tensor', max_fpr: 'float' = 1.0, *, "
        "standardized: 'bool' = False, mask: 'Tensor | None' = None) -> 'Tensor'"
    ),
}


def test_all_is_pinned():
    assert sorted(aucreshape.__all__) == sorted(EXPECTED_ALL)
    for name in aucreshape.__all__:
        assert hasattr(aucreshape, name)


@pytest.mark.parametrize("func", list(EXPECTED_SIGNATURES), ids=lambda f: f.__qualname__)
def test_signature_is_pinned(func):
    assert str(inspect.signature(func)) == EXPECTED_SIGNATURES[func]


@pytest.mark.parametrize(
    "name",
    [
        "target_specificity",
        "num_labels",
        "strategy",
        "min_negatives",
        "capacity",
        "thresholds",
        "ready",
    ],
)
def test_tracker_read_only_properties(name):
    assert isinstance(getattr(aucreshape.ThresholdTracker, name), property)


def test_loss_thresholds_property():
    assert isinstance(aucreshape.AUCReshapingLoss.thresholds, property)


def test_public_metric_module_all():
    assert sorted(metrics.__all__) == [
        "partial_auc",
        "sensitivity_at_specificity",
        "sensitivity_at_threshold",
        "threshold_at_specificity",
    ]
