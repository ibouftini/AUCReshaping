"""Tests for benchmarks.review.arms: controls must reproduce AUCReshapingLoss exactly."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from aucreshape import AUCReshapingLoss, metrics
from benchmarks.review import arms

SPECS = [0.9, 0.95]
BOOSTS = {"prob": [0.3, 0.1], "weight": [2.0, 1.0], "logit": [2.0, 1.0]}


def batches(n: int = 3, size: int = 64) -> list[tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        y = (torch.rand(size, generator=gen) < 0.4).float()
        z = torch.randn(size, generator=gen) * 2 + y
        out.append((z, y))
    return out


@pytest.mark.parametrize("mode", ["prob", "weight", "logit"])
@pytest.mark.parametrize("combine", ["band", "sum"])
def test_threshold_rule_matches_the_library(mode: str, combine: str) -> None:
    common = {
        "boost": BOOSTS[mode],
        "mode": mode,
        "combine": combine,
        "pos_weight": 2.0,
        "initial_thresholds": [0.0, 1.0],
        "min_negatives": 8,
    }
    reference = AUCReshapingLoss(SPECS, **common)
    control = arms.ControlLoss("threshold", SPECS, **common)
    for z, y in batches():
        z1, z2 = z.clone().requires_grad_(), z.clone().requires_grad_()
        expected, got = reference(z1, y), control(z2, y)
        expected.backward()
        got.backward()
        torch.testing.assert_close(got, expected)
        torch.testing.assert_close(z2.grad, z1.grad)


def test_random_rule_boosts_as_many_positives_elsewhere() -> None:
    kwargs = {"boost": 0.2, "initial_thresholds": [1.0], "min_negatives": 8}
    rule = arms.ControlLoss("threshold", 0.9, **kwargs)
    rand = arms.ControlLoss("random", 0.9, seed=1, **kwargs)
    z, y = batches(1, 256)[0]
    rule(z.clone().requires_grad_(), y)
    rand(z.clone().requires_grad_(), y)
    assert rand.last_boosted.sum() == rule.last_boosted.sum() > 0
    assert not torch.equal(rand.last_boosted, rule.last_boosted)
    assert not (rand.last_boosted & (y < 0.5)).any()


def test_all_rule_boosts_every_positive_once_ready() -> None:
    loss = arms.ControlLoss("all", 0.9, boost=0.2, min_negatives=100)
    z, y = batches(1, 64)[0]
    loss(z.clone().requires_grad_(), y)  # 64 samples: fewer than 100 negatives, not ready
    assert not loss.last_boosted.any()
    for z, y in batches(3, 128):
        loss(z.clone().requires_grad_(), y)
    torch.testing.assert_close(loss.last_boosted, y > 0.5)


def test_above_rule_boosts_only_positives_above_the_threshold() -> None:
    loss = arms.ControlLoss("above", 0.9, boost=0.2, initial_thresholds=[0.5], min_negatives=8)
    z, y = batches(1, 256)[0]
    loss(z.clone().requires_grad_(), y)
    boosted = loss.last_boosted
    assert boosted.any()
    assert torch.all(z[boosted] >= 0.5)
    assert torch.equal(boosted, (y > 0.5) & (z >= 0.5))


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("ce", {}),
        ("weighted_bce", {"pos_weight": 4.0}),
        ("focal", {"gamma": 2.0}),
        ("aucreshape", {"boost": 0.2}),
        ("random_boost", {"boost": 0.2}),
        ("boost_all", {"boost": 0.2}),
        ("placebo", {"boost": 0.2}),
        ("heldout", {"boost": 0.2}),
        ("near_miss", {"boost": 0.2}),
        ("aucreshape_pw", {"boost": 0.2, "pos_weight": 4.0}),
        ("warm_aucreshape", {"boost": 0.2, "warmup_steps": 10}),
        ("hard_negatives", {"boost": 2.0}),
        ("smoothed", {"boost": 0.2, "momentum": 0.9}),
    ],
)
def test_make_arm_builds_every_arm(name: str, params: dict[str, float]) -> None:
    loss = arms.make_arm(name, params, target=0.9, pos_weight=None, seed=0, data_len=64)
    z, y = batches(1)[0]
    if getattr(loss, "needs_index", False):
        value = loss(z.clone().requires_grad_(), y, torch.arange(len(z)))
    else:
        value = loss(z.clone().requires_grad_(), y)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        arms.make_arm("nope", {}, target=0.9, pos_weight=None, seed=0)


@pytest.mark.parametrize("name", ["aucreshape", "random_boost", "boost_all", "placebo"])
def test_outside_training_every_reshaping_arm_is_bce(name: str) -> None:
    loss = arms.make_arm(name, {"boost": 0.4}, target=0.9, pos_weight=None, seed=0)
    for z, y in batches(5, 128):  # make the trackers ready
        loss(z.clone().requires_grad_(), y).backward()
    z, y = batches(1)[0]
    with torch.no_grad():
        torch.testing.assert_close(loss(z, y), nn.functional.binary_cross_entropy_with_logits(z, y))


def test_heldout_arm_takes_its_threshold_from_validation() -> None:
    loss = arms.make_arm("heldout", {"boost": 0.2}, target=0.9, pos_weight=None, seed=0)
    val_z, val_y = batches(1, 512)[0]
    arms.calibrate(loss, val_z, val_y)
    expected = metrics.threshold_at_specificity(val_z, val_y, 0.9)
    torch.testing.assert_close(loss.tracker.thresholds.flatten(), expected.flatten().float())


# ---- Interventions -----------------------------------------------


def test_near_miss_skips_positives_far_below_the_threshold() -> None:
    params = {
        "boost": 0.2,
        "initial_thresholds": [0.0, 1.0],
        "min_negatives": 8,
        "reduction": "none",
    }
    loss = arms.make_arm("near_miss", params, target=0.98, pos_weight=None, seed=0)
    z = torch.tensor([-1.0, 0.5, 2.0, 0.3], requires_grad=True)  # far below, between, above, neg
    y = torch.tensor([1.0, 1.0, 1.0, 0.0])
    got = loss(z, y)
    got.sum().backward()
    bce = nn.functional.binary_cross_entropy_with_logits(z.detach(), y, reduction="none")
    torch.testing.assert_close(got[[0, 2, 3]].detach(), bce[[0, 2, 3]])
    assert got[1] > bce[1]


def test_class_weighted_aucreshape_equals_the_library() -> None:
    params = {"boost": 0.2, "pos_weight": 4.0, "initial_thresholds": [0.0], "min_negatives": 8}
    ours = arms.make_arm("aucreshape_pw", params, target=0.9, pos_weight=None, seed=0)
    reference = AUCReshapingLoss(0.9, boost=0.2, pos_weight=4.0, initial_thresholds=[0.0],
                                 min_negatives=8)  # fmt: skip
    for z, y in batches():
        a, b = ours(z.clone().requires_grad_(), y), reference(z.clone().requires_grad_(), y)
        torch.testing.assert_close(a, b)
        a.backward()
        b.backward()


def test_warm_aucreshape_is_bce_during_warm_up() -> None:
    params = {"boost": 0.4, "warmup_steps": 2, "initial_thresholds": [1.0], "min_negatives": 8}
    loss = arms.make_arm("warm_aucreshape", params, target=0.9, pos_weight=None, seed=0)
    for step, (z, y) in enumerate(batches(3)):
        value = loss(z.clone().requires_grad_(), y)
        bce = nn.functional.binary_cross_entropy_with_logits(z, y)
        if step < 2:
            torch.testing.assert_close(value.detach(), bce)
        else:
            assert value > bce
        value.backward()


def test_hard_negatives_boosts_only_negatives_above_the_threshold() -> None:
    loss = arms.ControlLoss("negatives", 0.9, boost=2.0, mode="weight", initial_thresholds=[0.5],
                            min_negatives=8)  # fmt: skip
    z, y = batches(1, 256)[0]
    zz = z.clone().requires_grad_()
    value = loss(zz, y)
    value.backward()
    above = (y < 0.5) & (z >= 0.5)
    assert above.any()
    assert torch.equal(loss.last_boosted, above)
    weight = torch.where(above, 3.0, 1.0)
    expected = -(y * nn.functional.logsigmoid(z) + weight * (1 - y) * nn.functional.logsigmoid(-z))
    torch.testing.assert_close(value.detach(), expected.mean())


def test_smoothed_boost_is_a_per_sample_moving_average() -> None:
    loss = arms.SmoothedLoss(0.9, boost=0.2, momentum=0.5, data_len=6, initial_thresholds=[0.0],
                             min_negatives=8)  # fmt: skip
    assert loss.needs_index
    z = torch.tensor([-1.0, 2.0, -0.5, 1.0])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    index = torch.tensor([0, 1, 2, 3])
    loss(z.clone().requires_grad_(), y, index).backward()
    torch.testing.assert_close(loss.ema[:4], torch.tensor([0.1, 0.0, 0.0, 0.0]))
    loss(z.clone().requires_grad_(), y, index).backward()
    torch.testing.assert_close(loss.ema[:4], torch.tensor([0.15, 0.0, 0.0, 0.0]))
    assert loss.ema[4:].abs().sum() == 0  # samples not in the batch keep their value


def test_boosted_fraction_is_over_positives_for_controls_too() -> None:
    # The independent review found controls reported over all elements and AUCReshaping over
    # positives, so the column could not be compared across arms.
    logits = torch.tensor([-3.0, -2.0, 1.0, 2.0, 0.0, 0.5, -1.0, 1.5, 0.2, -0.5])
    targets = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    loss = arms.ControlLoss("random", 0.9, boost=0.2, initial_thresholds=[0.0], min_negatives=5)
    loss(logits.clone().requires_grad_(), targets)
    assert loss.last_boosted.any()
    fraction = arms.boosted_fraction(loss)
    assert fraction == pytest.approx(float(loss.last_boosted.float().sum()) / 4)
