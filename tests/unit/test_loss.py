"""Tests for AUCReshapingLoss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from aucreshape import AUCReshapingLoss, ThresholdTracker, auc_reshaping_loss, linear_ramp


def batch(n=64, c=None, seed=0, pos_rate=0.5):
    g = torch.Generator().manual_seed(seed)
    shape = (n,) if c is None else (n, c)
    z = torch.randn(shape, generator=g) * 2
    y = (torch.rand(shape, generator=g) < pos_rate).float()
    return z, y


def test_module_equals_functional_with_fixed_thresholds():
    z, y = batch()
    criterion = AUCReshapingLoss(
        [0.9, 0.95], boost=[0.2, 0.1], strategy="fixed", initial_thresholds=[0.0, 0.5]
    )
    out = criterion(z.requires_grad_(), y)
    ref = auc_reshaping_loss(z, y, torch.tensor([0.0, 0.5]), [0.2, 0.1])
    torch.testing.assert_close(out, ref)


def test_toy_batch_boosted_loss_goes_up_rest_unchanged():
    criterion = AUCReshapingLoss(0.9, boost=0.2, initial_thresholds=[0.0], reduction="none")
    logits = torch.tensor([2.0, -1.0, 0.5, -2.0], requires_grad=True)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = criterion(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert loss[1] > bce[1]
    torch.testing.assert_close(loss[[0, 2, 3]], bce[[0, 2, 3]])


def test_last_stats_shapes_and_values():
    criterion = AUCReshapingLoss(
        [0.9, 0.95], boost=[0.3, 0.1], num_labels=2, initial_thresholds=[0.0, 1.0]
    )
    z = torch.tensor([[-1.0, 2.0], [0.5, 0.5], [3.0, -3.0], [0.0, 0.0]], requires_grad=True)
    y = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    criterion(z, y)
    stats = criterion.last_stats
    assert stats["boosted_fraction"].shape == (2,)
    assert stats["below_fraction"].shape == (2, 2)
    assert stats["thresholds"].shape == (2, 2)
    assert stats["ready"].tolist() == [True, True]
    assert stats["step"].item() == 1
    # Label 0: positives -1, 0.5, 3 -> -1 is below 0 (and 1), 0.5 below 1 only.
    torch.testing.assert_close(stats["boosted_fraction"], torch.tensor([2 / 3, 2 / 3]))
    torch.testing.assert_close(stats["below_fraction"][0], torch.tensor([1 / 3, 2 / 3]))
    torch.testing.assert_close(stats["mean_boost"][0], torch.tensor((0.3 + 0.1) / 2))
    assert all(not v.requires_grad for v in stats.values())


@pytest.mark.parametrize("context", ["eval", "no_grad"])
def test_outside_training_returns_bce_and_changes_nothing(context):
    z, y = batch()
    criterion = AUCReshapingLoss(0.9, boost=0.4, initial_thresholds=[1.0], min_negatives=5)
    before = {
        k: v.clone() for k, v in criterion.state_dict().items() if isinstance(v, torch.Tensor)
    }
    if context == "eval":
        criterion.eval()
        out = criterion(z, y)
    else:
        with torch.no_grad():
            out = criterion(z, y)
    torch.testing.assert_close(out, F.binary_cross_entropy_with_logits(z, y))
    after = criterion.state_dict()
    for key, value in before.items():
        torch.testing.assert_close(after[key], value)


def test_update_order_batch_does_not_boost_itself():
    criterion = AUCReshapingLoss(0.5, boost=0.3, min_negatives=4)
    z = torch.tensor([-1.0, 0.0, 1.0, 2.0, -3.0, -2.0], requires_grad=True)
    y = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    first = criterion(z, y)
    # Not ready before the first batch: plain BCE, even though the batch makes it ready.
    torch.testing.assert_close(first, F.binary_cross_entropy_with_logits(z, y))
    assert criterion.tracker.ready.item()
    second = criterion(z, y)
    assert second > first


def test_auto_update_modes():
    z, y = batch()
    forced = AUCReshapingLoss(0.9, min_negatives=5, auto_update=True)
    with torch.no_grad():
        forced(z, y)
    assert forced.tracker.ready.item()
    assert forced.step.item() == 0  # step only counts training calls

    never = AUCReshapingLoss(0.9, min_negatives=5, auto_update=False)
    never(z.requires_grad_(), y)
    assert not never.tracker.ready.item()
    assert never.step.item() == 1
    never.update(z, y)
    assert never.tracker.ready.item()


def test_warmup_steps_delay_boosting():
    z, y = batch()
    criterion = AUCReshapingLoss(0.9, boost=0.3, initial_thresholds=[1.0], warmup_steps=2)
    bce = F.binary_cross_entropy_with_logits(z, y)
    zz = z.clone().requires_grad_()
    torch.testing.assert_close(criterion(zz, y), bce)
    torch.testing.assert_close(criterion(zz, y), bce)
    assert criterion(zz, y) > bce
    assert criterion.step.item() == 3


def test_warmup_with_fixed_strategy_ends():
    z, y = batch()
    criterion = AUCReshapingLoss(
        0.9, boost=0.3, strategy="fixed", initial_thresholds=[1.0], warmup_steps=1
    )
    zz = z.clone().requires_grad_()
    first = criterion(zz, y)
    assert criterion(zz, y) > first


def test_tracker_argument_defaults_and_conflicts():
    tracker = ThresholdTracker(
        [0.9, 0.98], num_labels=3, strategy="fixed", initial_thresholds=[0.0, 1.0]
    )
    criterion = AUCReshapingLoss(boost=[0.2, 0.1], tracker=tracker)
    assert criterion.tracker is tracker
    assert criterion.thresholds.shape == (3, 2)
    AUCReshapingLoss([0.9, 0.98], boost=0.1, num_labels=3, tracker=tracker)  # consistent: ok
    with pytest.raises(ValueError, match="differs from the tracker"):
        AUCReshapingLoss(0.95, tracker=tracker)
    with pytest.raises(ValueError, match="num_labels=2 differs"):
        AUCReshapingLoss(num_labels=2, tracker=tracker)
    with pytest.raises(ValueError, match=r"tracker arguments \['window_size'\]"):
        AUCReshapingLoss(tracker=tracker, window_size=10)
    with pytest.raises(ValueError, match="initial_thresholds"):
        AUCReshapingLoss(tracker=tracker, initial_thresholds=[0.0, 1.0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"boost": [0.1, 0.2]}, "one value per target_specificity"),
        ({"boost": -0.1}, "non-negative"),
        ({"boost": torch.zeros(3)}, "boost tensor"),
        ({"target_specificity": [0.9, 0.95], "boost": linear_ramp(1, 1)}, "exactly one"),
        ({"mode": "x"}, "mode"),
        ({"combine": "x"}, "combine"),
        ({"reduction": "x"}, "reduction"),
        ({"min_ratio": 1.5}, "min_ratio"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"target_specificity": 1.0}, r"\(0, 1\)"),
        ({"target_specificity": 98}, r"Did you mean 0\.98"),
        ({"target_specificity": [0.95, 0.9]}, "ascending"),
        ({"strategy": "fixed"}, "from_scores"),
        ({"strategy": "median"}, "strategy"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AUCReshapingLoss(**kwargs)


def test_num_labels_mismatch_on_forward():
    criterion = AUCReshapingLoss(0.9)
    with pytest.raises(ValueError, match="pass num_labels=4"):
        criterion(torch.zeros(8, 4), torch.zeros(8, 4))
    with pytest.raises(ValueError, match="pass num_labels=4"):
        criterion.update(torch.zeros(8, 4), torch.zeros(8, 4))


def test_tensor_and_schedule_boosts():
    z, y = batch(c=2)
    table = torch.tensor([[0.2, 0.1], [0.4, 0.3]])
    criterion = AUCReshapingLoss(
        [0.9, 0.95], boost=table, num_labels=2, initial_thresholds=[0.0, 0.5], combine="sum"
    )
    out = criterion(z.clone().requires_grad_(), y)
    ref = auc_reshaping_loss(z, y, torch.tensor([0.0, 0.5]), table, combine="sum")
    torch.testing.assert_close(out, ref)

    ramp = linear_ramp(0.2, 0.5)
    sched = AUCReshapingLoss(0.9, boost=ramp, num_labels=2, initial_thresholds=[0.5])
    out = sched(z.clone().requires_grad_(), y)
    torch.testing.assert_close(out, auc_reshaping_loss(z, y, torch.tensor([0.5]), ramp))
    scalar_tensor = AUCReshapingLoss(0.9, boost=torch.tensor(0.2), initial_thresholds=[0.0])
    zb, yb = batch()
    torch.testing.assert_close(
        scalar_tensor(zb.clone().requires_grad_(), yb),
        auc_reshaping_loss(zb, yb, torch.tensor([0.0]), 0.2),
    )


@pytest.mark.parametrize("pos_weight", [2.0, torch.tensor([2.0, 0.5])])
def test_pos_weight_and_weight_match_bce_without_boost(pos_weight):
    z, y = batch(c=2)
    w = torch.tensor([1.0, 3.0])
    criterion = AUCReshapingLoss(0.9, num_labels=2, pos_weight=pos_weight, weight=w).eval()
    pw = pos_weight if isinstance(pos_weight, torch.Tensor) else torch.tensor([2.0, 2.0])
    torch.testing.assert_close(
        criterion(z, y), F.binary_cross_entropy_with_logits(z, y, weight=w, pos_weight=pw)
    )
    assert "pos_weight" not in criterion.state_dict()


def test_mask_and_nan_targets_in_training():
    z, y = batch(c=3)
    y[::2, 1] = float("nan")
    mask = ~torch.isnan(y)
    criterion = AUCReshapingLoss(0.9, num_labels=3, initial_thresholds=[0.5], min_negatives=2)
    zz = z.clone().requires_grad_()
    loss = criterion(zz, y, mask=mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert zz.grad is not None
    assert torch.isfinite(zz.grad).all()
    assert (zz.grad[~mask] == 0).all()


@pytest.mark.parametrize("shape", [(16,), (16, 1)])
def test_binary_shapes(shape):
    z = torch.randn(shape, requires_grad=True)
    y = (torch.rand(shape) < 0.5).float()
    criterion = AUCReshapingLoss(0.9, reduction="none", initial_thresholds=[0.0])
    assert criterion(z, y).shape == shape


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_logits_give_float32_loss(dtype):
    z, y = batch()
    criterion = AUCReshapingLoss(0.9, initial_thresholds=[0.0])
    zz = z.to(dtype).requires_grad_()
    out = criterion(zz, y)
    assert out.dtype == torch.float32
    out.backward()
    assert zz.grad is not None
    assert zz.grad.dtype == dtype


def test_state_dict_round_trip_gives_identical_continuation():
    torch.manual_seed(0)
    data = [batch(seed=s) for s in range(6)]
    a = AUCReshapingLoss(0.9, boost=0.2, min_negatives=20, window_size=64)
    for z, y in data[:3]:
        a(z.clone().requires_grad_(), y)
    state = a.state_dict()
    b = AUCReshapingLoss(0.9, boost=0.2, min_negatives=20, window_size=64)
    b.load_state_dict(state)
    for z, y in data[3:]:
        la = a(z.clone().requires_grad_(), y)
        lb = b(z.clone().requires_grad_(), y)
        torch.testing.assert_close(la, lb)
    torch.testing.assert_close(a.thresholds, b.thresholds)
    assert a.step.item() == b.step.item() == 6


def test_load_state_dict_with_other_config_raises():
    a = AUCReshapingLoss(0.9, window_size=64, min_negatives=20)
    b = AUCReshapingLoss(0.95, window_size=64, min_negatives=20)
    with pytest.raises(ValueError, match="different configuration"):
        b.load_state_dict(a.state_dict())


def test_end_epoch_delegates_and_repr():
    criterion = AUCReshapingLoss(0.9, strategy="epoch", min_negatives=5)
    z, y = batch()
    criterion(z.requires_grad_(), y)
    assert not criterion.tracker.ready.item()
    criterion.end_epoch()
    assert criterion.tracker.ready.item()
    assert "mode='prob'" in repr(criterion)
    assert "strategy='epoch'" in repr(criterion)
    assert "schedule" in repr(AUCReshapingLoss(0.9, boost=linear_ramp(1, 1)))


def test_empty_batch_in_training_is_zero():
    criterion = AUCReshapingLoss(0.9, initial_thresholds=[0.0])
    z = torch.zeros(0, requires_grad=True)
    out = criterion(z, torch.zeros(0))
    assert out.item() == 0.0
    out.backward()


def test_disabled_label_threshold_never_boosts():
    tracker = ThresholdTracker(0.9, strategy="fixed", initial_thresholds=[-math.inf])
    criterion = AUCReshapingLoss(boost=0.4, tracker=tracker)
    z, y = batch()
    zz = z.clone().requires_grad_()
    torch.testing.assert_close(criterion(zz, y), F.binary_cross_entropy_with_logits(z, y))


@pytest.mark.gpu
def test_module_follows_cuda_device():
    criterion = AUCReshapingLoss(0.9, min_negatives=5)
    z, y = batch()
    out = criterion(z.cuda().requires_grad_(), y.cuda())
    assert out.device.type == "cuda"
    assert criterion.tracker.estimate.device.type == "cuda"


def test_warns_when_training_loss_is_discarded_without_backward():
    """A validation pass with gradients enabled silently updates the tracker; flag it."""
    criterion = AUCReshapingLoss(0.9, min_negatives=5)
    z, y = batch()
    loss = criterion(z.clone().requires_grad_(), y)  # e.g. validation without no_grad
    with pytest.warns(UserWarning, match=r"without backward\(\).*torch.no_grad"):
        del loss


def test_no_warning_for_backward_or_summed_accumulation(recwarn):
    import gc

    criterion = AUCReshapingLoss(0.9, min_negatives=5)
    z, y = batch()
    criterion(z.clone().requires_grad_(), y).backward()
    total = sum(criterion(z.clone().requires_grad_(), y) for _ in range(3))  # one backward
    total.backward()
    del total
    with torch.no_grad():
        criterion(z, y)
    gc.collect()
    assert not [w for w in recwarn if "backward()" in str(w.message)]
