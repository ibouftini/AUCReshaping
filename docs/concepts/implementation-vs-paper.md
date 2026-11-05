# The implementation and the paper

This page lists every choice this library makes where the paper leaves room, so that you can judge
whether a result reflects the method or our reading of it. It is written from the paper alone: a
search on 2026-11-05 found no public code from the authors, and the paper has no code statement.

Page numbers refer to the 16-page PDF of Bhat et al., *Scientific Reports* 13, 21097 (2023).

## Naming

**"AUCReshaping" alone means the paper's form**: `mode="prob"`, Eq. 1. Every other mode or variant
is named as such in tables, figures and text. This is a correction: in the first version of our
[Results](../results.md), the row named "AUCReshaping" could be the `weight` variant.

## The loss

| Decision | The paper | Our choice | Why | Test | Varied in an experiment? |
|---|---|---|---|---|---|
| Loss | Eq. 1 (p. 4): $-\sum_i [y_i \log(p_i - b_i) + (1 - y_i)\log(1 - p_i + b_i)]$, with $b_i = n$ if $y_i = 1$ and $p_i < \theta_{max}$ | The same, with `mode="prob"`, one threshold and the default `combine` | This is the method's definition | `test_exact_eq1_single_threshold` | Not applicable |
| $p_i - b_i \le 0$ | Not discussed. Table 3 (p. 9) uses a boost of 1 and Table 5 (p. 10) a boost of 0.8, so Eq. 1 is undefined for every boosted sample at a boost of 1 | A detached cap: $p - b \ge r\,p$, with `min_ratio` $r = 0.1$. The gradient stays between 1 and 10 times that of BCE | A plain clamp gives hard positives no gradient; the literal value has no definition. See [Numerics](numerics.md) | `test_hard_positive_keeps_nonzero_gradient`, `test_prob_gradient_bound` | **Not yet in a benchmark.** It matters: with the paper's boosts on one saved chest X-ray model, $r$ of 0.1, 0.25 and 0.5 give about 7.0, 3.3 and 1.9 times the gradient of BCE. The new study varies it |
| Reduction | Eq. 1 is a sum | Mean by default; `reduction="sum"` is available | It matches `BCEWithLogitsLoss`, so swapping losses keeps the learning rate meaningful | `test_zero_boost_equals_bce` | No |
| Comparison | Strict, $p_i < \theta_{max}$ (p. 4) | Strict | As stated | `test_only_misclassified_positives_change` | No |
| $\theta$ in probability or logit space | Probability (p. 4) | Stored as logits | Equivalent, because the sigmoid is monotone, and numerically safer | `test_probability_and_logit_scores_agree_on_sensitivity` | No |
| `prob`, `logit`, `weight` modes | Eq. 1 is a probability offset. Boosts "modify the loss values or output logits" (p. 8) | `prob` is the paper's form. `logit` and `weight` are variants, named as such | The paper's sentence allows more than one implementation, and its large boosts suggest it may have used one | `test_weight_mode_equals_weights_times_bce`, `test_loss_worked_table` | Yes: in two earlier benchmarks |

## Several thresholds and boost sizes

| Decision | The paper | Our choice | Why | Test | Varied in an experiment? |
|---|---|---|---|---|---|
| How boosts combine | Boosts "can be applied at various thresholds" (p. 4). "[0.4, 0.2, 0.1, 0.1] at [0.90, 0.92, 0.95, 0.98]" means "0.4 is applied at the 0.90 threshold" and so on (p. 8). It does not say what a sample below several thresholds receives | `combine="band"` by default (the boost of the lowest threshold it falls below), with `sum` and `max` available | `band` fits decreasing boosts. Constant boosts, such as "0.2 at 0.90, 0.92, 0.95, 0.98" (Table 3, p. 9), make little sense under it, because it reduces them to one threshold; `sum` suits those | `test_boost_values_worked_table` | Only in one small study. **Open:** our earlier credit-card arm used `band` with a constant boost. On saved credit-card models, the same printed configuration gives about 1.5 times the gradient of BCE under `band` and 9.8 times under `sum`. The new study tests both |
| Size of the boost | "A piece-wise approximation of the average distance" of misclassified positives from the boundary (p. 4), and "currently implemented only as a constant boosting value" (p. 4) | A constant you choose, or a schedule of the distance below $\theta$ (`linear_ramp`) | The paper gives values, not the rule behind them. Its adaptive boost is future work (p. 5) | `test_step_schedule_reproduces_band_single_threshold` | Tuned over a grid. The adaptive boost is not implemented |

## The threshold

| Decision | The paper | Our choice | Why | Test | Varied in an experiment? |
|---|---|---|---|---|---|
| Source and window | "In each iteration" (Fig. 3, p. 5), and "the corresponding thresholds from the ROC curve during training" (p. 8). It does not say whether this means one batch, one epoch or something else | A first-in, first-out window of recent training scores of label 0, in training mode (with any dropout), updated at every step. `batch`, `ema`, `epoch` and `fixed` are available | One batch holds too few reference samples at 98% specificity to set a threshold | `test_window_matches_quantile_of_last_negatives`, `test_window_batching_invariance` | Yes: other strategies, and thresholds from validation or from the true distribution, changed little in synthetic data |
| At test time | The threshold found on validation is "carried over to the testing phase" (p. 3). Test sensitivity is also read from the test ROC (p. 5) | Both are available; `metrics` reads the ROC | The two readings can differ widely, so results should show both | `test_sensitivity_at_specificity_interpolated` (against scikit-learn), `test_from_scores_matches_metrics_and_calibrate_refreshes` | Reported separately in the new study |

## When boosting starts, and how it interacts with training

| Decision | The paper | Our choice | Why | Test | Varied in an experiment? |
|---|---|---|---|---|---|
| Stage | "Implemented during the fine-tuning stage" (p. 4) of a pre-trained model (p. 3) | The library does not enforce a stage | It is a loss function, usable in any loop | `test_warmup_steps_delay_boosting` | **Partly.** The main synthetic arms trained from random initialization, unlike the paper; only one arm started with CE. The new study adds a fine-tuning start |
| First step | Not discussed | Boosting starts once the tracker holds $\lceil 10 / (1 - s) \rceil$ reference samples. `warmup_steps` can delay it further (default 0) | A quantile from fewer samples is too noisy | `test_default_min_negatives`, `test_not_ready_until_min_negatives` | Yes: one synthetic arm |
| Class weights | Class weights from the class ratio (p. 8), and "positive_weights of 5" for credit card (p. 8). It does not say whether the boost comes before the weight | `pos_weight` scales the boosted positive term, as in `BCEWithLogitsLoss` | The closest match to the code the paper names | `test_pos_weight_scales_boosted_positive_term`, `test_pos_weight_and_weight_match_bce_without_boost` | Yes: one synthetic arm |
| Samplers | "Distributed weighted sampling" over-samples the negative classes (p. 8) | Left to you; the loss does not need to know | A sampler belongs to the data loader | Not applicable | **No.** The first chest X-ray study had no sampler, because it is not defined for many labels at once |
| Multi-label | Each label's ROC is handled independently (p. 12) | The same, with per-label thresholds and boosts | As stated | `test_multilabel_equals_independent_binary` | Used in the first chest X-ray study |
| Multi-class | Sketched for BI-RADS 5 (p. 12) | Not implemented | The sketch is not specific enough to reproduce | Not applicable | No |

## Labels

| Decision | The paper | Our choice | Why | Test | Varied in an experiment? |
|---|---|---|---|---|---|
| Which class is boosted | "Samples with the lower misclassification cost" (p. 4). Normal is label 1: "1=absent, 0=present" (p. 5), and 227,454 positive training samples for credit card (p. 8). See also [Label conventions](label-conventions.md) | Label 1 is the class whose sensitivity you raise; specificity is held on label 0. Both conventions work unchanged | The mathematics does not depend on the names | `test_paper_convention_is_the_mirror_image` | **Not yet on real images.** On chest X-rays, normals were always boosted. Disease = 1 was only measured. The new study adds boosting disease |
| Mammography | Positive is BI-RADS 4 and 5 (p. 7), but BI-RADS 1 and 2 elsewhere (p. 13) | Not applicable | Noted so that nobody relies on either | Not applicable | Not run |

## Where our first evaluation departed from the paper

An independent review, given only the paper and the code, looked for ways the code could disadvantage
the method. It found the core loss, thresholds and metrics correct, and these problems in how the first
evaluation was set up. The new protocol addresses each one.

- **One shared model.** The first chest X-ray study trained 15 outputs together, where the paper trains
  one binary task at a time, so the boosted task received only part of the training signal.
- **A different "abnormal" task.** It was any finding against no finding. The paper's combined task is
  effusion or pneumothorax against everything else, so it cannot be compared with the paper's Table 5.
- **The random control** differed in strength as well as targeting. See [Results](../results.md).
- **The synthetic scenarios** had no class weights, boosted the rare class and started from random
  initialization, unlike the paper. Even the gentlest setting tried was already a strong boost.

## Where the paper is inconsistent

These do not change the method, but they affect any comparison with it:

- **Table 7 and its text.** The table reads sensitivity "@0.98 Specificity" and the text (p. 12) says
  99% for the same 0.73. Our baseline agrees better with 98%, though this cannot prove it.
- **Table 2.** The credit-card sizes sum to 284,847, which is 40 more than the dataset. The training
  size on p. 8 (227,454 + 391 = 227,845) is the consistent one.
- **Fig. 2.** Its boxes say "at 98% sensitivity", while the text everywhere says specificity. We follow
  the text.
- **"Top 5 models were averaged"** (p. 8). The meaning is unclear. We average every run, because
  selecting runs by result would inflate any gain.
