"""Tests for ThresholdTracker."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aucreshape import ThresholdTracker, metrics


def stream(n=4000, seed=0, pos_rate=0.3, c=None):
    g = torch.Generator().manual_seed(seed)
    shape = (n,) if c is None else (n, c)
    y = (torch.rand(shape, generator=g) < pos_rate).float()
    z = torch.randn(shape, generator=g) + 1.5 * y
    return z, y


def feed(tracker, z, y, batch_size, mask=None):
    for i in range(0, z.shape[0], batch_size):
        m = None if mask is None else mask[i : i + batch_size]
        tracker.update(z[i : i + batch_size], y[i : i + batch_size], m)


SPECS = [0.9, 0.95, 0.98]


# --- Quantile correctness at steady state -----------------------------------------------------


def test_window_matches_quantile_of_last_negatives():
    z, y = stream()
    tracker = ThresholdTracker(SPECS, window_size=1000)
    feed(tracker, z, y, 64)
    negatives = z[y == 0][-1000:]
    expected = torch.quantile(negatives, torch.tensor(SPECS))
    torch.testing.assert_close(tracker.thresholds[0], expected)
    assert tracker.support.item() == 1000
    assert tracker.ready.item()


def test_batch_strategy_is_quantile_of_latest_batch():
    z, y = stream(n=3000)
    tracker = ThresholdTracker(SPECS, strategy="batch", min_negatives=100)
    feed(tracker, z, y, 1000)
    last = z[2000:][y[2000:] == 0]
    torch.testing.assert_close(tracker.thresholds[0], torch.quantile(last, torch.tensor(SPECS)))


def test_epoch_strategy_equals_full_quantile_when_reservoir_is_large():
    z, y = stream()
    tracker = ThresholdTracker(SPECS, strategy="epoch", max_samples=10_000)
    feed(tracker, z, y, 100)
    assert not tracker.ready.item()  # nothing until the epoch is finalised
    tracker.end_epoch()
    expected = torch.quantile(z[y == 0], torch.tensor(SPECS))
    torch.testing.assert_close(tracker.thresholds[0], expected)
    assert tracker.ready.item()


def test_ema_update_formula():
    tracker = ThresholdTracker(0.5, strategy="ema", momentum=0.75, min_negatives=1)
    tracker.update(torch.tensor([0.0, 2.0]), torch.zeros(2))  # median 1 -> initialises
    assert tracker.thresholds.item() == pytest.approx(1.0)
    tracker.update(torch.tensor([4.0, 6.0]), torch.zeros(2))  # median 5
    assert tracker.thresholds.item() == pytest.approx(0.75 * 1.0 + 0.25 * 5.0)
    assert tracker.support.item() == 4


def test_ema_with_zero_momentum_is_batch():
    z, y = stream(n=2000)
    a = ThresholdTracker(SPECS, strategy="ema", momentum=0.0)
    b = ThresholdTracker(SPECS, strategy="batch", min_negatives=1)
    feed(a, z, y, 500)
    feed(b, z, y, 500)
    torch.testing.assert_close(a.estimate, b.estimate)


def test_streaming_estimates_close_to_metric_threshold():
    z, y = stream(n=20_000, seed=3)
    tracker = ThresholdTracker(SPECS, strategy="epoch", max_samples=20_000)
    feed(tracker, z, y, 256)
    tracker.end_epoch()
    exact = metrics.threshold_at_specificity(z, y, SPECS)
    torch.testing.assert_close(tracker.thresholds[0], exact, atol=0.02, rtol=0)


# --- Window FIFO --------------------------------------------------------------------------------


def test_window_fifo_overwrites_oldest_first():
    tracker = ThresholdTracker(0.5, window_size=5, min_negatives=1)
    tracker.update(torch.arange(3.0), torch.zeros(3))  # 0 1 2
    tracker.update(torch.arange(3.0, 8.0), torch.zeros(5))  # keeps 3..7
    kept = torch.sort(tracker.buffer[0, :5]).values
    torch.testing.assert_close(kept, torch.arange(3.0, 8.0))
    assert tracker.thresholds.item() == pytest.approx(5.0)


def test_window_batch_larger_than_capacity_keeps_last_values():
    tracker = ThresholdTracker(0.5, window_size=4, min_negatives=1)
    tracker.update(torch.arange(10.0), torch.zeros(10))
    torch.testing.assert_close(torch.sort(tracker.buffer[0, :4]).values, torch.arange(6.0, 10.0))
    assert tracker.support.item() == 4


def test_window_batching_invariance():
    z, y = stream(n=1500)
    a = ThresholdTracker(SPECS, window_size=700)
    b = ThresholdTracker(SPECS, window_size=700)
    feed(a, z, y, 37)
    feed(b, z, y, 500)
    torch.testing.assert_close(a.thresholds, b.thresholds)


# --- Reservoir ---------------------------------------------------------------------------------


def reservoir_after(values, cap, batch_size, seed=0):
    tracker = ThresholdTracker(0.5, strategy="epoch", max_samples=cap, min_negatives=1, seed=seed)
    feed(tracker, values, torch.zeros_like(values), batch_size)
    return tracker.buffer[0, :cap].clone()


def test_reservoir_is_uniform():
    values = torch.arange(20_000, dtype=torch.float32)
    kept = reservoir_after(values, cap=2000, batch_size=500)
    counts = np.histogram(kept.numpy(), bins=10, range=(0, 20_000))[0]
    expected = 200
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert chi2 < 27.9  # 99.9% quantile of chi-square with 9 dof
    assert len(set(kept.tolist())) == 2000  # sampling without replacement


def test_reservoir_is_deterministic_and_batching_invariant():
    values = torch.arange(5000, dtype=torch.float32)
    a = reservoir_after(values, 300, 64, seed=1)
    b = reservoir_after(values, 300, 1000, seed=1)
    c = reservoir_after(values, 300, 64, seed=2)
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, c)


def test_end_epoch_twice_is_harmless_and_clears_reservoir():
    z, y = stream(n=2000)
    tracker = ThresholdTracker(0.9, strategy="epoch", min_negatives=50)
    feed(tracker, z, y, 100)
    tracker.end_epoch()
    first = tracker.thresholds.clone()
    tracker.end_epoch()
    torch.testing.assert_close(tracker.thresholds, first)
    assert tracker.buffer_count.item() == 0
    assert tracker.total_seen.item() == 0


# --- Warm-up and readiness -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "needed"), [(0.95, 200), (0.98, 500), (0.99, 1000), (0.999, 10000)]
)
def test_default_min_negatives(spec, needed):
    assert ThresholdTracker(spec).min_negatives == needed
    assert ThresholdTracker(spec).capacity == max(4096, 2 * needed)
    assert ThresholdTracker(spec, strategy="epoch").capacity == max(65536, 2 * needed)


def test_not_ready_until_min_negatives():
    tracker = ThresholdTracker(0.95)  # needs 200 negatives
    tracker.update(torch.randn(199), torch.zeros(199))
    assert not tracker.ready.item()
    assert torch.isinf(tracker.active_thresholds()).all()
    tracker.update(torch.randn(1), torch.zeros(1))
    assert tracker.ready.item()
    assert torch.isfinite(tracker.active_thresholds()).all()


def test_initial_thresholds_warm_start_then_switch():
    tracker = ThresholdTracker(0.5, initial_thresholds=[7.0], min_negatives=10)
    assert tracker.ready.item()
    assert tracker.thresholds.item() == 7.0
    tracker.update(torch.zeros(5), torch.zeros(5))
    assert tracker.thresholds.item() == 7.0  # own estimate not valid yet
    tracker.update(torch.zeros(5), torch.zeros(5))
    assert tracker.thresholds.item() == 0.0


def test_capacity_errors():
    with pytest.raises(ValueError, match="window_size=100 is smaller than min_negatives=200"):
        ThresholdTracker(0.95, window_size=100)
    with pytest.raises(ValueError, match="max_samples"):
        ThresholdTracker(0.95, strategy="epoch", max_samples=10)
    with pytest.raises(ValueError, match="positive integer"):
        ThresholdTracker(0.95, window_size=0)


def test_warns_once_when_not_ready_at_end_epoch():
    tracker = ThresholdTracker(0.99)
    tracker.update(torch.randn(10), torch.zeros(10))
    with pytest.warns(UserWarning, match=r"not ready at the end of the epoch.*1000"):
        tracker.end_epoch()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracker.end_epoch()  # only once
    tracker.reset()
    with pytest.warns(UserWarning, match="not ready"):
        tracker.end_epoch()  # re-armed by reset


def test_warns_after_many_updates_without_readiness():
    tracker = ThresholdTracker(0.99, strategy="batch")
    for _ in range(999):
        tracker.update(torch.randn(8), torch.zeros(8))
    with pytest.warns(UserWarning, match=r"after 1000 updates.*batches need"):
        tracker.update(torch.randn(8), torch.zeros(8))


# --- Edge cases ---------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["batch", "window", "ema", "epoch"])
def test_batch_without_negatives_leaves_state_unchanged(strategy):
    tracker = ThresholdTracker(0.5, strategy=strategy, min_negatives=1)
    tracker.update(torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3))
    before = {k: v.clone() for k, v in tracker.state_dict().items() if isinstance(v, torch.Tensor)}
    tracker.update(torch.tensor([9.0, 9.0]), torch.ones(2))
    tracker.update(torch.zeros(0), torch.zeros(0))
    for key, value in tracker.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, before[key])


def test_mask_half_targets_and_non_finite_logits_are_skipped():
    tracker = ThresholdTracker(0.5, min_negatives=1)
    z = torch.tensor([1.0, 100.0, float("nan"), float("inf"), 3.0, 50.0])
    y = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    mask = torch.tensor([True, False, True, True, True, True])
    tracker.update(z, y, mask)
    assert tracker.support.item() == 2
    assert tracker.thresholds.item() == pytest.approx(2.0)


def test_single_sample_and_ties():
    tracker = ThresholdTracker(0.9, min_negatives=1)
    tracker.update(torch.tensor([0.3]), torch.tensor([0.0]))
    assert tracker.thresholds.item() == pytest.approx(0.3)
    tracker.update(torch.full((5,), 0.3), torch.zeros(5))
    assert tracker.thresholds.item() == pytest.approx(0.3)


# --- Fixed, calibrate, from_scores --------------------------------------------------------------


def test_fixed_requires_initial_and_never_updates():
    with pytest.raises(ValueError, match="from_scores"):
        ThresholdTracker(0.9, strategy="fixed")
    tracker = ThresholdTracker([0.9, 0.95], strategy="fixed", initial_thresholds=[0.0, 1.0])
    tracker.update(torch.randn(100), torch.zeros(100))
    torch.testing.assert_close(tracker.thresholds, torch.tensor([[0.0, 1.0]]))
    assert tracker.ready.item()


def test_from_scores_matches_metrics_and_calibrate_refreshes():
    z, y = stream(n=5000, c=2)
    tracker = ThresholdTracker.from_scores(z, y, [0.9, 0.95])
    expected = metrics.threshold_at_specificity(z, y, [0.9, 0.95])
    torch.testing.assert_close(tracker.thresholds, expected)
    assert tracker.strategy == "fixed"
    z2, y2 = stream(n=5000, c=2, seed=9)
    tracker.calibrate(z2, y2)
    torch.testing.assert_close(
        tracker.thresholds, metrics.threshold_at_specificity(z2, y2, [0.9, 0.95])
    )


def test_calibrate_warns_and_handles_missing_negatives():
    z = torch.randn(30, 2)
    y = torch.zeros(30, 2)
    y[:, 1] = 1.0  # label 1 has no negatives
    with pytest.warns(UserWarning, match=r"label\(s\) \[0, 1\] have fewer"):
        tracker = ThresholdTracker.from_scores(z, y, 0.98)
    assert math.isinf(tracker.thresholds[1, 0].item())
    assert tracker.active_thresholds()[1, 0].item() == -math.inf


def test_calibrate_warm_starts_streaming_strategy():
    z, y = stream(n=3000)
    tracker = ThresholdTracker(0.95)
    tracker.calibrate(z, y)
    assert tracker.ready.item()
    torch.testing.assert_close(
        tracker.thresholds[0, 0], metrics.threshold_at_specificity(z, y, 0.95)
    )


# --- Multi-label independence -----------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:ThresholdTracker is not ready:UserWarning")  # batch strategy
@pytest.mark.parametrize("strategy", ["window", "epoch", "ema", "batch"])
def test_multilabel_equals_independent_binary_trackers(strategy):
    z, y = stream(n=3000, c=3, seed=5)
    mask = torch.rand(3000, 3, generator=torch.Generator().manual_seed(1)) < 0.8
    multi = ThresholdTracker(SPECS, num_labels=3, strategy=strategy, min_negatives=50)
    feed(multi, z, y, 128, mask)
    multi.end_epoch()
    for c in range(3):
        single = ThresholdTracker(SPECS, strategy=strategy, min_negatives=50)
        feed(single, z[:, c], y[:, c], 128, mask[:, c])
        single.end_epoch()
        if strategy == "epoch":
            continue  # reservoir draws are keyed by label, so only the sizes match
        torch.testing.assert_close(multi.thresholds[c], single.thresholds[0])
        assert multi.support[c] == single.support[0]


# --- State ------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["window", "epoch", "ema", "batch"])
def test_state_dict_round_trip(strategy):
    z, y = stream(n=4000)
    a = ThresholdTracker(SPECS, strategy=strategy, min_negatives=50, seed=3)
    feed(a, z[:2000], y[:2000], 100)
    b = ThresholdTracker(SPECS, strategy=strategy, min_negatives=50, seed=3)
    b.load_state_dict(a.state_dict())
    feed(a, z[2000:], y[2000:], 100)
    feed(b, z[2000:], y[2000:], 100)
    a.end_epoch()
    b.end_epoch()
    torch.testing.assert_close(a.thresholds, b.thresholds)
    for key, value in a.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(b.state_dict()[key], value)


def test_load_state_dict_config_mismatch():
    a = ThresholdTracker(0.9, window_size=500)
    with pytest.raises(ValueError, match=r"target_specificity: saved \[0.9\], here \[0.95\]"):
        ThresholdTracker(0.95, window_size=500).load_state_dict(a.state_dict())
    with pytest.raises(ValueError, match="strategy"):
        ThresholdTracker(0.9, strategy="ema").load_state_dict(a.state_dict(), strict=False)


def test_reset_keeps_initial_thresholds():
    tracker = ThresholdTracker(0.5, initial_thresholds=[2.0], min_negatives=1)
    tracker.update(torch.zeros(4), torch.zeros(4))
    assert tracker.thresholds.item() == 0.0
    tracker.reset()
    assert tracker.thresholds.item() == 2.0
    assert tracker.support.item() == 0


def test_buffers_survive_half_cast():
    tracker = ThresholdTracker(0.5, min_negatives=1).half()
    tracker.update(torch.tensor([1.0, 3.0]), torch.zeros(2))
    assert tracker.thresholds.dtype == torch.float32
    assert tracker.thresholds.item() == pytest.approx(2.0)


# --- Validation and repr ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"strategy": "median"}, "strategy"),
        ({"momentum": 1.0}, "momentum"),
        ({"num_labels": 0}, "num_labels"),
        ({"min_negatives": 0}, "min_negatives"),
        ({"initial_thresholds": [0.0, 1.0]}, "initial_thresholds"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ThresholdTracker(0.9, **kwargs)


def test_label_mismatch_errors():
    tracker = ThresholdTracker(0.9, num_labels=2)
    with pytest.raises(ValueError, match="num_labels=3"):
        tracker.update(torch.zeros(4, 3), torch.zeros(4, 3))
    with pytest.raises(ValueError, match="num_labels=3"):
        tracker.calibrate(torch.zeros(4, 3), torch.zeros(4, 3))


def test_repr_and_properties():
    tracker = ThresholdTracker([0.9, 0.99], num_labels=2, window_size=2000)
    text = repr(tracker)
    assert "target_specificity=(0.9, 0.99)" in text
    assert "capacity=2000" in text
    assert tracker.target_specificity == (0.9, 0.99)
    assert tracker.num_labels == 2
    assert tracker.strategy == "window"


@pytest.mark.gpu
def test_tracker_follows_cuda():
    tracker = ThresholdTracker(0.9, min_negatives=5)
    tracker.update(torch.randn(50).cuda(), torch.zeros(50).cuda())
    assert tracker.buffer.device.type == "cuda"
    assert tracker.ready.item()
