"""Tests for aucreshape.functional, including a worked numeric example."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from aucreshape import functional as AF

# --- worked example -----------------------------------------------------------------------

Z = [2.0, 0.5, -1.0, 0.5]
Y = [1.0, 1.0, 1.0, 0.0]
THETA = [0.0, 1.0]
BOOSTS = [0.1, 0.3]
BCE = [0.126928, 0.474077, 1.313262, 0.974077]
TABLE_B = {"band": [0.0, 0.3, 0.1, 0.0], "sum": [0.0, 0.3, 0.4, 0.0], "max": [0.0, 0.3, 0.3, 0.0]}
TABLE_LOSS = {
    ("band", "prob"): [BCE[0], 1.131778, 1.778203, BCE[3]],
    ("band", "logit"): [BCE[0], 0.598139, 1.387335, BCE[3]],
    ("band", "weight"): [BCE[0], 0.616300, 1.444588, BCE[3]],
    ("sum", "prob"): [BCE[0], 1.131778, 3.615847, BCE[3]],
    ("sum", "logit"): [BCE[0], 0.598139, 1.620417, BCE[3]],
    ("sum", "weight"): [BCE[0], 0.616300, 1.838566, BCE[3]],
    ("max", "prob"): [BCE[0], 1.131778, 3.615847, BCE[3]],
    ("max", "logit"): [BCE[0], 0.598139, 1.541008, BCE[3]],
    ("max", "weight"): [BCE[0], 0.616300, 1.707240, BCE[3]],
}
MODES = ["prob", "logit", "weight"]
COMBINES = ["band", "sum", "max"]


def table_inputs(dtype=torch.float64):
    return (
        torch.tensor(Z, dtype=dtype),
        torch.tensor(Y, dtype=dtype),
        torch.tensor(THETA, dtype=dtype),
    )


@pytest.mark.parametrize("combine", COMBINES)
def test_boost_values_worked_table(combine):
    z, y, theta = table_inputs()
    b = AF.boost_values(z, y, theta, BOOSTS, combine=combine)
    torch.testing.assert_close(b, torch.tensor(TABLE_B[combine], dtype=torch.float64))
    assert not b.requires_grad


@pytest.mark.parametrize("combine", COMBINES)
@pytest.mark.parametrize("mode", MODES)
def test_loss_worked_table(combine, mode):
    z, y, theta = table_inputs()
    loss = AF.auc_reshaping_loss(z, y, theta, BOOSTS, mode=mode, combine=combine, reduction="none")
    expected = torch.tensor(TABLE_LOSS[(combine, mode)], dtype=torch.float64)
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=0)


def test_prob_gradient_worked_table_hits_bound():
    z, y, theta = table_inputs()
    z.requires_grad_(True)
    AF.auc_reshaping_loss(z, y, theta, BOOSTS, combine="max", reduction="sum").backward()
    assert z.grad is not None
    assert z.grad[2].item() == pytest.approx(-7.310586, abs=1e-6)


# --- Faithfulness to Eq. 1 ------------------------------------------------------------------


def eq1_reference(z, y, b):
    """Direct float64 Eq. 1 of the paper (b only on positives)."""
    p = torch.sigmoid(z.double())
    y = y.double()
    return -(y * torch.log(p - b) + (1 - y) * torch.log(1 - p + b))


def test_exact_eq1_single_threshold():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(512, generator=g, dtype=torch.float64) * 2
    y = (torch.rand(512, generator=g) < 0.5).double()
    theta = torch.tensor([0.5], dtype=torch.float64)
    n = 0.02
    b = torch.where((y == 1) & (z < 0.5), torch.tensor(n, dtype=torch.float64), 0.0)
    p = torch.sigmoid(z)
    # Eq. 1 is reproduced wherever the floor is inactive (b <= (1 - min_ratio) * p).
    active = b <= 0.9 * p
    assert active.float().mean() > 0.9
    loss = AF.auc_reshaping_loss(z, y, theta, n, reduction="none")
    torch.testing.assert_close(loss[active], eq1_reference(z, y, b)[active])


# --- Reduces to BCE -------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("pos_weight", [None, 2.5])
@pytest.mark.parametrize("mode", MODES)
def test_zero_boost_equals_bce(dtype, reduction, pos_weight, mode):
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, 3, generator=g).to(dtype)
    y = (torch.rand(64, 3, generator=g) < 0.3).to(dtype)
    pw = None if pos_weight is None else torch.full((3,), pos_weight, dtype=dtype)
    loss = AF.auc_reshaping_loss(
        z, y, torch.tensor([0.0, 1.0]), 0.0, mode=mode, pos_weight=pw, reduction=reduction
    )
    ref = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction=reduction)
    torch.testing.assert_close(loss, ref)
    assert loss.dtype == dtype


@pytest.mark.parametrize("mode", MODES)
def test_disabled_thresholds_equal_bce(mode):
    z = torch.randn(32)
    y = (torch.rand(32) < 0.5).float()
    theta = torch.tensor([-math.inf, -math.inf])
    loss = AF.auc_reshaping_loss(z, y, theta, [0.4, 0.2], mode=mode)
    torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(z, y))


# --- Locality -------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("combine", COMBINES)
def test_only_misclassified_positives_change(mode, combine):
    g = torch.Generator().manual_seed(3)
    z = (torch.randn(200, generator=g, dtype=torch.float64) * 3).requires_grad_(True)
    y = (torch.rand(200, generator=g) < 0.5).double()
    theta = torch.tensor([-0.5, 0.5], dtype=torch.float64)
    loss = AF.auc_reshaping_loss(
        z, y, theta, [0.2, 0.1], mode=mode, combine=combine, reduction="none"
    )
    (grad,) = torch.autograd.grad(loss.sum(), z)
    z0 = z.detach().clone().requires_grad_(True)
    bce = F.binary_cross_entropy_with_logits(z0, y, reduction="none")
    (bce_grad,) = torch.autograd.grad(bce.sum(), z0)

    boosted = (y == 1) & (z.detach() < 0.5)
    torch.testing.assert_close(loss[~boosted], bce[~boosted])
    torch.testing.assert_close(grad[~boosted], bce_grad[~boosted])
    assert boosted.any()
    assert (loss[boosted] > bce[boosted]).all()
    # Boosted positives get a stronger (more negative) gradient pushing the logit up.
    assert (grad[boosted] < bce_grad[boosted]).all()


# --- Numerics -------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_logits_are_finite(mode, dtype):
    z = torch.tensor([-100.0, -30.0, -1.0, 0.0, 1.0, 30.0, 100.0] * 2, dtype=dtype)
    y = torch.tensor([1.0] * 7 + [0.0] * 7, dtype=dtype)
    z.requires_grad_(True)
    theta = torch.tensor([50.0, 200.0], dtype=dtype)  # every positive is below both
    loss = AF.auc_reshaping_loss(z, y, theta, [0.4, 0.9], mode=mode, combine="sum")
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_hard_positive_keeps_nonzero_gradient():
    """A naive clamp(p - b, eps) would zero this gradient (p << b)."""
    z = torch.tensor([-6.0], requires_grad=True)  # p ~ 0.0025, boost 0.4 >> p
    y = torch.tensor([1.0])
    loss = AF.auc_reshaping_loss(z, y, torch.tensor([0.0]), 0.4)
    loss.backward()
    assert z.grad is not None
    bce_grad = -(1 - torch.sigmoid(torch.tensor(-6.0)))
    # Exactly 1/min_ratio times the BCE gradient: finite, non-zero and bounded.
    assert z.grad.item() == pytest.approx(10 * bce_grad.item(), rel=1e-5)


@pytest.mark.parametrize("min_ratio", [0.05, 0.1, 0.5])
def test_prob_gradient_bound(min_ratio):
    z = torch.linspace(-20, 5, 101, dtype=torch.float64, requires_grad=True)
    y = torch.ones(101, dtype=torch.float64)
    loss = AF.auc_reshaping_loss(
        z, y, torch.tensor([10.0], dtype=torch.float64), 0.9, min_ratio=min_ratio, reduction="sum"
    )
    (grad,) = torch.autograd.grad(loss, z)
    bce_grad = -(1 - torch.sigmoid(z.detach()))
    ratio = grad / bce_grad
    assert (ratio >= 1 - 1e-9).all()
    assert (ratio <= 1 / min_ratio + 1e-9).all()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("combine", COMBINES)
def test_gradcheck(mode, combine):
    # Logits are kept away from the thresholds so the indicator is locally constant, and the
    # boosts are small enough that the prob-mode cap is inactive: the cap is detached by
    # design, so finite differences cannot see it (test_prob_gradient_bound covers it).
    z = torch.tensor([2.0, 0.5, -1.0, 0.5, -1.5, 1.7], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    theta = torch.tensor([0.0, 1.0], dtype=torch.float64)

    def fn(x):
        return AF.auc_reshaping_loss(x, y, theta, [0.05, 0.1], mode=mode, combine=combine)

    assert torch.autograd.gradcheck(fn, (z,))


def test_nan_logits_propagate():
    z = torch.tensor([0.0, float("nan")])
    y = torch.tensor([1.0, 0.0])
    loss = AF.auc_reshaping_loss(z, y, torch.tensor([1.0]), 0.1)
    assert torch.isnan(loss)


# --- Shapes, reductions, masks --------------------------------------------------------------


@pytest.mark.parametrize(
    ("logit_shape", "target_shape"),
    [((8,), (8,)), ((8, 1), (8, 1)), ((8, 1), (8,)), ((8,), (8, 1)), ((8, 4), (8, 4))],
)
def test_shapes(logit_shape, target_shape):
    z = torch.randn(logit_shape)
    y = (torch.rand(target_shape) < 0.5).float()
    out = AF.auc_reshaping_loss(z, y, torch.tensor([0.0]), 0.1, reduction="none")
    assert out.shape == z.shape
    assert AF.boost_values(z, y, torch.tensor([0.0]), 0.1).shape == z.shape
    assert AF.auc_reshaping_loss(z, y, torch.tensor([0.0]), 0.1).shape == ()


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_empty_batch_gives_zero(reduction):
    z = torch.zeros(0, requires_grad=True)
    loss = AF.auc_reshaping_loss(z, torch.zeros(0), torch.tensor([0.0]), 0.1, reduction=reduction)
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None
    assert z.grad.shape == (0,)


def test_empty_batch_none_reduction():
    out = AF.auc_reshaping_loss(
        torch.zeros(0, 3), torch.zeros(0, 3), torch.tensor([0.0]), 0.1, reduction="none"
    )
    assert out.shape == (0, 3)


def test_mask_excludes_elements_and_denominator():
    z = torch.tensor([[0.3, -2.0], [1.0, 0.2], [-1.0, 4.0]])
    y = torch.tensor([[1.0, float("nan")], [0.0, 1.0], [1.0, float("nan")]])
    mask = ~torch.isnan(y)
    z.requires_grad_(True)
    loss = AF.auc_reshaping_loss(z, y, torch.tensor([0.5]), 0.1, mask=mask)
    per = AF.auc_reshaping_loss(z, y, torch.tensor([0.5]), 0.1, mask=mask, reduction="none")
    torch.testing.assert_close(loss, per[mask].sum() / mask.sum())
    assert (per[~mask] == 0).all()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()  # NaN targets under the mask are harmless
    assert (z.grad[~mask] == 0).all()


def test_fully_masked_batch_gives_zero():
    z = torch.randn(4, requires_grad=True)
    loss = AF.auc_reshaping_loss(
        z, torch.ones(4), torch.tensor([9.0]), 0.1, mask=torch.zeros(4, dtype=torch.bool)
    )
    assert loss.item() == 0.0


def test_masked_positive_is_not_boosted():
    z = torch.tensor([-1.0, -1.0])
    y = torch.tensor([1.0, 1.0])
    b = AF.boost_values(z, y, torch.tensor([0.0]), 0.2, mask=torch.tensor([True, False]))
    torch.testing.assert_close(b, torch.tensor([0.2, 0.0]))


def test_soft_targets_positive_iff_above_half():
    z = torch.full((4,), -3.0)
    y = torch.tensor([0.9, 0.6, 0.5, 0.1])
    b = AF.boost_values(z, y, torch.tensor([0.0]), 0.2)
    torch.testing.assert_close(b, torch.tensor([0.2, 0.2, 0.0, 0.0]))


def test_per_label_thresholds_and_boosts():
    z = torch.tensor([[-1.0, -1.0], [0.5, 0.5]])
    y = torch.ones(2, 2)
    theta = torch.tensor([[0.0, 1.0], [-2.0, 0.0]])  # (C, K)
    boosts = torch.tensor([[0.1, 0.3], [0.5, 0.7]])
    b = AF.boost_values(z, y, theta, boosts, combine="sum")
    torch.testing.assert_close(b, torch.tensor([[0.4, 0.7], [0.3, 0.0]]))


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_weight_matches_bce_weight(reduction):
    z = torch.randn(16, 3)
    y = (torch.rand(16, 3) < 0.5).float()
    w = torch.rand(3)
    out = AF.auc_reshaping_loss(z, y, torch.tensor([-math.inf]), 0.3, weight=w, reduction=reduction)
    ref = F.binary_cross_entropy_with_logits(z, y, weight=w, reduction=reduction)
    torch.testing.assert_close(out, ref)


def test_weight_on_binary_logits():
    z = torch.randn(8)
    y = torch.ones(8)
    w = torch.rand(8)
    out = AF.auc_reshaping_loss(z, y, torch.tensor([0.0]), 0.1, weight=w, reduction="none")
    plain = AF.auc_reshaping_loss(z, y, torch.tensor([0.0]), 0.1, reduction="none")
    torch.testing.assert_close(out, w * plain)


def test_pos_weight_scales_boosted_positive_term():
    z = torch.tensor([-1.0], dtype=torch.float64)
    y = torch.tensor([1.0], dtype=torch.float64)
    theta = torch.tensor([0.0], dtype=torch.float64)
    plain = AF.auc_reshaping_loss(z, y, theta, 0.1)
    weighted = AF.auc_reshaping_loss(z, y, theta, 0.1, pos_weight=3.0)
    torch.testing.assert_close(weighted, 3 * plain)


# --- dtype / device -------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
def test_half_inputs_are_computed_in_float32(dtype, mode):
    z = torch.randn(64).to(dtype).requires_grad_(True)
    y = (torch.rand(64) < 0.5).to(dtype)
    loss = AF.auc_reshaping_loss(z, y, torch.tensor([0.5]), 0.3, mode=mode)
    ref = AF.auc_reshaping_loss(z.float(), y.float(), torch.tensor([0.5]), 0.3, mode=mode)
    assert loss.dtype == torch.float32  # computed and returned in float32, like BCE + autocast
    torch.testing.assert_close(loss, ref)
    loss.backward()
    assert z.grad is not None
    assert z.grad.dtype == dtype
    assert torch.isfinite(z.grad.float()).all()


@pytest.mark.gpu
def test_cuda_matches_cpu():
    z = torch.randn(128)
    y = (torch.rand(128) < 0.5).float()
    theta = torch.tensor([0.0, 0.5])
    cpu = AF.auc_reshaping_loss(z, y, theta, [0.2, 0.1])
    gpu = AF.auc_reshaping_loss(z.cuda(), y.cuda(), theta.cuda(), [0.2, 0.1])
    torch.testing.assert_close(gpu.cpu(), cpu)


def test_thresholds_on_other_dtype_are_moved():
    z = torch.randn(8, dtype=torch.float64)
    y = torch.ones(8, dtype=torch.float64)
    out = AF.auc_reshaping_loss(z, y, torch.tensor([0.0], dtype=torch.float32), 0.1)
    assert out.dtype == torch.float64


# --- Weights and schedules ------------------------------------------------------------------


@pytest.mark.parametrize("combine", COMBINES)
def test_weight_mode_equals_weights_times_bce(combine):
    z = torch.randn(50, 2)
    y = (torch.rand(50, 2) < 0.5).float()
    theta = torch.tensor([-0.3, 0.4])
    w = AF.reshaping_weights(z, y, theta, [0.5, 0.25], combine=combine)
    torch.testing.assert_close(w, 1 + AF.boost_values(z, y, theta, [0.5, 0.25], combine=combine))
    loss = AF.auc_reshaping_loss(
        z, y, theta, [0.5, 0.25], mode="weight", combine=combine, reduction="none"
    )
    torch.testing.assert_close(loss, w * F.binary_cross_entropy_with_logits(z, y, reduction="none"))
    assert (w >= 1).all()


def test_linear_ramp_values_and_monotonicity():
    ramp = AF.linear_ramp(0.5, 0.4)
    d = torch.linspace(0, 3, 31)
    b = ramp(d)
    torch.testing.assert_close(b, torch.clamp(0.5 * d, max=0.4))
    assert (b.diff() >= 0).all()


def test_schedule_boost_uses_distance_below_threshold():
    z = torch.tensor([1.5, 0.5, -1.0, -1.0])
    y = torch.tensor([1.0, 1.0, 1.0, 0.0])
    b = AF.boost_values(z, y, torch.tensor([1.0]), AF.linear_ramp(0.1, 1.0))
    torch.testing.assert_close(b, torch.tensor([0.0, 0.05, 0.2, 0.0]))


def test_step_schedule_reproduces_band_single_threshold():
    z = torch.randn(100)
    y = (torch.rand(100) < 0.5).float()
    theta = torch.tensor([0.2])

    def step(d):
        return torch.where(d > 0, torch.full_like(d, 0.3), torch.zeros_like(d))

    torch.testing.assert_close(
        AF.boost_values(z, y, theta, step), AF.boost_values(z, y, theta, 0.3, combine="band")
    )


def test_schedule_negative_outputs_are_clamped():
    b = AF.boost_values(
        torch.tensor([-1.0]), torch.tensor([1.0]), torch.tensor([0.0]), lambda d: -d
    )
    assert b.item() == 0.0


# --- Validation errors ----------------------------------------------------------------------

GOOD_Z = torch.zeros(4)
GOOD_Y = torch.zeros(4)
GOOD_T = torch.tensor([0.0])


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"logits": torch.zeros(4, dtype=torch.long)}, TypeError, "floating"),
        ({"logits": torch.zeros(2, 2, 2), "targets": torch.zeros(2, 2, 2)}, ValueError, "1 or 2"),
        ({"targets": torch.zeros(5)}, ValueError, "same shape"),
        ({"mask": torch.ones(3)}, ValueError, "mask"),
        ({"thresholds": torch.zeros(2, 2, 2)}, ValueError, r"\(K,\) or \(C, K\)"),
        ({"thresholds": torch.zeros(3, 1)}, ValueError, "labels"),
        ({"thresholds": torch.zeros(0)}, ValueError, "at least one"),
        ({"boost": [0.1, 0.2]}, ValueError, "one boost per threshold"),
        ({"boost": -0.1}, ValueError, "non-negative"),
        ({"boost": [0.1, float("nan")], "thresholds": torch.zeros(2)}, ValueError, "non-neg"),
        ({"boost": torch.zeros(3)}, ValueError, "boost must have shape"),
        ({"combine": "avg"}, ValueError, "band"),
        ({"mode": "margin"}, ValueError, "prob"),
        ({"reduction": "avg"}, ValueError, "mean"),
        ({"min_ratio": 0.0}, ValueError, "min_ratio"),
        ({"min_ratio": 1.0}, ValueError, "min_ratio"),
        ({"pos_weight": torch.ones(3)}, ValueError, "pos_weight"),
        ({"boost": AF.linear_ramp(1, 1), "thresholds": torch.zeros(2)}, ValueError, "one"),
    ],
)
def test_validation_errors(kwargs, exc, match):
    args = {"logits": GOOD_Z, "targets": GOOD_Y, "thresholds": GOOD_T, "boost": 0.1}
    args.update(kwargs)
    with pytest.raises(exc, match=match):
        AF.auc_reshaping_loss(**args)


def test_linear_ramp_validation():
    with pytest.raises(ValueError, match="slope"):
        AF.linear_ramp(-1.0, 1.0)
    with pytest.raises(ValueError, match="max_boost"):
        AF.linear_ramp(1.0, -1.0)


def test_per_label_boosts_with_shared_thresholds():
    z = torch.tensor([[-1.0, -1.0]])
    y = torch.ones(1, 2)
    b = AF.boost_values(z, y, torch.tensor([0.0]), torch.tensor([[0.1], [0.4]]))
    torch.testing.assert_close(b, torch.tensor([[0.1, 0.4]]))


def test_reshaping_weights_compose_with_focal_loss():
    """The weights plug into any per-sample loss; here a local sigmoid focal loss."""

    def focal(logits, targets, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (1 - torch.exp(-bce)) ** gamma * bce

    z = torch.randn(64, requires_grad=True)
    y = (torch.rand(64) < 0.5).float()
    theta = torch.tensor([0.3])
    w = AF.reshaping_weights(z, y, theta, 1.0)
    loss = (w * focal(z, y)).mean()
    loss.backward()
    assert z.grad is not None
    boosted = (y == 1) & (z.detach() < 0.3)
    assert boosted.any()
    torch.testing.assert_close(w[boosted], torch.full_like(w[boosted], 2.0))
    torch.testing.assert_close(w[~boosted], torch.ones_like(w[~boosted]))
    assert not w.requires_grad  # weights are constants for autograd
