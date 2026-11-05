"""Tests for aucreshape.metrics against scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from aucreshape import metrics

SPECS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.999]


def make_data(seed, n=300, ties=False, pos_rate=0.3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(np.int64)
    s = rng.normal(size=n) + 1.2 * y
    if ties:
        s = np.round(s, 1)
    return s, y


def sk_threshold(s, y, spec):
    fpr, _, thr = roc_curve(y, s, drop_intermediate=False)
    return thr[fpr <= 1 - spec + 1e-9][-1]


def sk_sens_interp(s, y, spec):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    # Same 1e-9 tolerance as the implementation: 1 - 0.8 must hit a vertex at FPR 0.2.
    return np.interp(1 - spec + 1e-9, fpr, tpr)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ties", [False, True])
def test_threshold_matches_sklearn(seed, ties):
    s, y = make_data(seed, ties=ties)
    out = metrics.threshold_at_specificity(torch.tensor(s), torch.tensor(y), SPECS)
    expected = [sk_threshold(s, y, v) for v in SPECS]
    np.testing.assert_allclose(out.numpy(), expected)
    # The threshold really achieves the specificity on this data.
    neg = s[y == 0]
    for tau, v in zip(out.tolist(), SPECS, strict=True):
        assert (neg < tau).mean() >= v - 1e-12


def test_threshold_tiny_set_and_tolerance():
    # 10 negatives at specificity 0.9 allow exactly one false positive.
    s = torch.arange(10, dtype=torch.float64)
    y = torch.zeros(10)
    assert metrics.threshold_at_specificity(s, y, 0.9).item() == 9.0
    # Only one negative: a specificity of 0.5 needs zero false positives -> +inf.
    assert metrics.threshold_at_specificity(torch.tensor([1.0]), torch.tensor([0]), 0.5) == float(
        "inf"
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ties", [False, True])
def test_sensitivity_at_specificity_interpolated(seed, ties):
    s, y = make_data(seed, ties=ties)
    out = metrics.sensitivity_at_specificity(torch.tensor(s), torch.tensor(y), SPECS)
    expected = [sk_sens_interp(s, y, v) for v in SPECS]
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)


@pytest.mark.parametrize("seed", range(3))
def test_sensitivity_at_specificity_achievable(seed):
    s, y = make_data(seed)
    out = metrics.sensitivity_at_specificity(
        torch.tensor(s), torch.tensor(y), SPECS, interpolate=False
    )
    expected = [(s[y == 1] >= sk_threshold(s, y, v)).mean() for v in SPECS]
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)
    interp = metrics.sensitivity_at_specificity(torch.tensor(s), torch.tensor(y), SPECS)
    assert (interp.numpy() >= out.numpy() - 1e-9).all()


def test_vertical_segment_takes_top():
    # Two positives tied with no negative: a vertical ROC segment at FPR 0.
    s = torch.tensor([3.0, 3.0, 1.0, 0.0])
    y = torch.tensor([1, 1, 0, 0])
    assert metrics.sensitivity_at_specificity(s, y, 0.99).item() == 1.0


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("max_fpr", [0.01, 0.05, 0.1, 0.3, 1.0])
@pytest.mark.parametrize("ties", [False, True])
def test_partial_auc_standardized_matches_sklearn(seed, max_fpr, ties):
    s, y = make_data(seed, ties=ties)
    out = metrics.partial_auc(torch.tensor(s), torch.tensor(y), max_fpr, standardized=True)
    expected = roc_auc_score(y, s, max_fpr=None if max_fpr == 1.0 else max_fpr)
    assert out.item() == pytest.approx(expected, abs=1e-9)


def test_partial_auc_raw_bounds():
    s, y = make_data(0)
    raw = metrics.partial_auc(torch.tensor(s), torch.tensor(y), 0.1)
    assert 0.0 <= raw.item() <= 0.1
    perfect = metrics.partial_auc(torch.tensor([0.0, 1.0]), torch.tensor([0, 1]), 0.1)
    assert perfect.item() == pytest.approx(0.1)


def test_full_auc_matches_sklearn():
    s, y = make_data(7, ties=True)
    assert metrics.partial_auc(torch.tensor(s), torch.tensor(y)).item() == pytest.approx(
        roc_auc_score(y, s)
    )


def test_sensitivity_at_threshold_values_and_shapes():
    s = torch.tensor([0.1, 0.4, 0.6, 0.9])
    y = torch.tensor([1, 1, 1, 0])
    assert metrics.sensitivity_at_threshold(s, y, 0.5).item() == pytest.approx(1 / 3)
    out = metrics.sensitivity_at_threshold(s, y, torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(out, torch.tensor([1.0, 1 / 3, 0.0]))
    assert metrics.sensitivity_at_threshold(s, y, torch.tensor([[0.0, 0.5]])).shape == (2,)


def test_multilabel_shapes_and_independence():
    s0, y0 = make_data(1)
    s1, y1 = make_data(2)
    s = torch.tensor(np.stack([s0, s1], 1))
    y = torch.tensor(np.stack([y0, y1], 1))
    thr = metrics.threshold_at_specificity(s, y, [0.9, 0.95])
    assert thr.shape == (2, 2)
    np.testing.assert_allclose(thr[1].numpy(), [sk_threshold(s1, y1, v) for v in (0.9, 0.95)])
    assert metrics.sensitivity_at_specificity(s, y, 0.9).shape == (2,)
    assert metrics.partial_auc(s, y, 0.1).shape == (2,)
    assert metrics.sensitivity_at_threshold(s, y, thr).shape == (2, 2)
    assert metrics.sensitivity_at_threshold(s, y, 0.0).shape == (2,)


def test_binary_n1_inputs_give_scalars():
    s, y = make_data(3)
    st = torch.tensor(s).unsqueeze(-1)
    assert metrics.threshold_at_specificity(st, torch.tensor(y), 0.9).shape == ()
    assert metrics.partial_auc(st, torch.tensor(y).unsqueeze(-1)).shape == ()


def test_mask_and_ignored_half_targets():
    s, y = make_data(4)
    keep = np.arange(len(s)) % 3 != 0
    full = metrics.partial_auc(torch.tensor(s[keep]), torch.tensor(y[keep]), 0.2)
    masked = metrics.partial_auc(torch.tensor(s), torch.tensor(y), 0.2, mask=torch.tensor(keep))
    assert masked.item() == pytest.approx(full.item())
    y_half = y.astype(float)
    y_half[~keep] = 0.5
    halves = metrics.partial_auc(torch.tensor(s), torch.tensor(y_half), 0.2)
    assert halves.item() == pytest.approx(full.item())


def test_missing_class_gives_nan_and_warns():
    s = torch.randn(10)
    with pytest.warns(UserWarning, match="no negative"):
        out = metrics.threshold_at_specificity(s, torch.ones(10), 0.9)
    assert torch.isnan(out)
    with pytest.warns(UserWarning, match="no positive"):
        assert torch.isnan(metrics.sensitivity_at_specificity(s, torch.zeros(10), 0.9))
    with pytest.warns(UserWarning, match="no positive"):
        assert torch.isnan(metrics.partial_auc(s, torch.zeros(10)))
    with pytest.warns(UserWarning, match="no positive"):
        assert torch.isnan(metrics.sensitivity_at_threshold(s, torch.zeros(10), 0.0))


def test_probability_and_logit_scores_agree_on_sensitivity():
    s, y = make_data(5)
    logits = torch.tensor(s)
    a = metrics.sensitivity_at_specificity(logits, torch.tensor(y), 0.95)
    b = metrics.sensitivity_at_specificity(torch.sigmoid(logits), torch.tensor(y), 0.95)
    assert a.item() == pytest.approx(b.item())


def test_output_dtype_and_device():
    s = torch.randn(20, dtype=torch.float16)
    y = torch.arange(20) % 2
    assert metrics.partial_auc(s, y).dtype == torch.float32
    assert metrics.partial_auc(s.double(), y).dtype == torch.float64


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda: metrics.threshold_at_specificity(torch.zeros(3), torch.zeros(3), 1.0),
            r"\(0, 1\)",
        ),
        (lambda: metrics.threshold_at_specificity(torch.zeros(3), torch.zeros(3), []), "at least"),
        (lambda: metrics.partial_auc(torch.zeros(3), torch.zeros(3), 0.0), "max_fpr"),
        (lambda: metrics.partial_auc(torch.zeros(3), torch.zeros(4)), "same shape"),
        (lambda: metrics.partial_auc(torch.zeros(2, 2, 2), torch.zeros(2, 2, 2)), r"\(N,\)"),
        (lambda: metrics.partial_auc(torch.zeros(3), torch.zeros(3), mask=torch.ones(2)), "mask"),
        (
            lambda: metrics.sensitivity_at_threshold(
                torch.zeros(3), torch.zeros(3), torch.zeros(2, 2)
            ),
            "binary",
        ),
        (
            lambda: metrics.sensitivity_at_threshold(
                torch.zeros(3, 2), torch.zeros(3, 2), torch.zeros(3)
            ),
            "labels",
        ),
    ],
)
def test_validation_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_percentage_specificity_gets_a_hint():
    with pytest.raises(ValueError, match=r"Did you mean 0\.95"):
        metrics.sensitivity_at_specificity(torch.zeros(3), torch.zeros(3), 95)


def test_rejects_non_tensors():
    with pytest.raises(TypeError, match="Tensor"):
        metrics.partial_auc([0.1, 0.2], torch.tensor([0, 1]))


def test_nan_targets_are_treated_as_missing():
    s = torch.arange(10, dtype=torch.float64)
    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, float("nan")], dtype=torch.float64)
    # 5 negatives at specificity 0.8 allow one false positive -> threshold 4. Counting the
    # NaN (score 9) as a negative would push it to 5.
    assert metrics.threshold_at_specificity(s, y, 0.8).item() == 4.0
    assert metrics.sensitivity_at_threshold(s, y, 5.0).item() == 1.0
