"""Property-based invariants (Hypothesis)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aucreshape import ThresholdTracker, auc_reshaping_loss, boost_values, reshaping_weights

MODES = ["prob", "logit", "weight"]
COMBINES = ["band", "sum", "max"]
settings.register_profile("aucreshape", max_examples=60, deadline=None, derandomize=True)
settings.load_profile("aucreshape")

finite = st.floats(-1e4, 1e4, allow_nan=False, allow_subnormal=False, width=32)


@st.composite
def problems(draw, max_labels=4, max_k=3):
    n = draw(st.integers(0, 40))
    c = draw(st.integers(1, max_labels))
    k = draw(st.integers(1, max_k))
    z = draw(hnp.arrays(np.float32, (n, c), elements=finite))
    y = draw(hnp.arrays(np.float32, (n, c), elements=st.sampled_from([0.0, 1.0])))
    theta = np.sort(draw(hnp.arrays(np.float32, (c, k), elements=st.floats(-20, 20, width=32))), 1)
    boost = draw(hnp.arrays(np.float32, (k,), elements=st.floats(0, 2, width=32)))
    mode = draw(st.sampled_from(MODES))
    combine = draw(st.sampled_from(COMBINES))
    t = torch.from_numpy
    return t(z), t(y), t(theta), t(boost), mode, combine


@given(problems())
def test_boosted_loss_never_below_bce_and_finite(problem):
    z, y, theta, boost, mode, combine = problem
    loss = auc_reshaping_loss(z, y, theta, boost, mode=mode, combine=combine, reduction="none")
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    assert torch.isfinite(loss).all()
    assert (loss >= bce - 1e-4 * (1 + bce.abs())).all()


@given(problems())
def test_negatives_and_correct_positives_are_never_boosted(problem):
    z, y, theta, boost, _, combine = problem
    b = boost_values(z, y, theta, boost, combine=combine)
    below_any = (z.unsqueeze(-1) < theta.unsqueeze(0)).any(-1)
    assert (b[y == 0] == 0).all()
    assert (b[(y == 1) & ~below_any] == 0).all()
    assert (b >= 0).all()


@given(problems())
def test_weights_at_least_one(problem):
    z, y, theta, boost, _, combine = problem
    assert (reshaping_weights(z, y, theta, boost, combine=combine) >= 1).all()


@given(problems(), st.randoms(use_true_random=False))
def test_permutation_invariance(problem, rnd):
    z, y, theta, boost, mode, combine = problem
    perm = list(range(len(z)))
    rnd.shuffle(perm)
    p = torch.tensor(perm, dtype=torch.long)
    kwargs = {"mode": mode, "combine": combine}
    a = auc_reshaping_loss(z, y, theta, boost, reduction="none", **kwargs)
    b = auc_reshaping_loss(z[p], y[p], theta, boost, reduction="none", **kwargs)
    torch.testing.assert_close(a[p], b)
    torch.testing.assert_close(
        auc_reshaping_loss(z, y, theta, boost, reduction="sum", **kwargs),
        auc_reshaping_loss(z[p], y[p], theta, boost, reduction="sum", **kwargs),
    )


@given(problems())
def test_multilabel_equals_independent_binary(problem):
    z, y, theta, boost, mode, combine = problem
    multi = auc_reshaping_loss(z, y, theta, boost, mode=mode, combine=combine, reduction="none")
    for c in range(z.shape[1]):
        single = auc_reshaping_loss(
            z[:, c], y[:, c], theta[c], boost, mode=mode, combine=combine, reduction="none"
        )
        torch.testing.assert_close(multi[:, c], single)


@given(
    hnp.arrays(
        np.float32,
        st.integers(1, 64),
        elements=st.floats(
            -3.0000000054977558e38, 3.0000000054977558e38, allow_subnormal=False, width=32
        ),
    )
)
def test_per_element_loss_finite_for_any_finite_logit(z_np):
    z = torch.from_numpy(z_np).requires_grad_()
    y = torch.ones_like(z)
    loss = auc_reshaping_loss(z, y, torch.tensor([1e30]), 0.9, reduction="none")
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


@given(
    hnp.arrays(np.float32, st.integers(30, 120), elements=st.floats(-5, 5, width=32)),
    st.floats(0.5, 0.95),
)
def test_paper_convention_is_the_mirror_image(scores, spec):
    """With fraud = 1, flipping logits and labels thresholds on fraud scores and boosts
    the normal transactions that a fraud-catching threshold would flag."""
    z = torch.from_numpy(scores)
    fraud = (torch.arange(len(z)) % 3 == 0).float()
    tracker = ThresholdTracker(spec, min_negatives=1)
    tracker.update(-z, 1 - fraud)  # paper convention: normal = 1, fraud is the reference
    theta = tracker.thresholds
    expected = torch.quantile(-z[fraud == 1], spec)
    torch.testing.assert_close(theta[0, 0], expected)
    b = boost_values(-z, 1 - fraud, theta, 0.3)
    torch.testing.assert_close(b > 0, (fraud == 0) & (-z < expected))
    assert math.isfinite(float(theta))
