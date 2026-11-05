# The method

## The problem

A single AUC number hides how a model performs at the threshold you deploy. Screening,
fraud and defect detection run at a **fixed high specificity** (few false alarms on the
reference class) and care about **sensitivity there**. That is one narrow region of the ROC
curve, and standard losses optimise the whole curve.

## The algorithm (paper, Eq. 1)

During fine-tuning, at every iteration:

1. find the threshold $\theta$ at which the negatives reach the target specificity (e.g. 0.98);
2. find the positives with predicted probability $p_i < \theta$ (the false negatives at that
   operating point);
3. boost their loss:

$$
\ell_i = -\big[\,y_i \log(p_i - b_i) + (1-y_i)\log(1-p_i)\big],\qquad
b_i = \begin{cases} n & y_i = 1 \text{ and } p_i < \theta \\ 0 & \text{otherwise}\end{cases}
$$

Because $-\log(p - n) > -\log p$, the boosted positives get a larger loss and a larger
gradient. Everything else is ordinary binary cross-entropy.

**Several thresholds** can carry different boosts. The paper's example is
`[0.4, 0.2, 0.1, 0.1]` at specificities `[0.90, 0.92, 0.95, 0.98]`: the further below the
operating point a positive is, the larger its boost. See
[Thresholds and boosts](thresholds-and-boosts.md).

## What this library adds to the paper

| Paper | `aucreshape` |
|---|---|
| Eq. 1 is undefined when $p \le b$ (the paper itself uses boosts up to 1.0) | a detached floor keeps it finite and the gradient alive ([Numerics](numerics.md)) |
| "threshold in each iteration" from the training ROC | a `ThresholdTracker` with five strategies, warm-up, DDP all-gather and checkpointing |
| boosts on "loss values or output logits" | `mode="prob"` (Eq. 1), `"logit"` (a margin), `"weight"` (a loss weight) |
| "discrete or continuous" boosts | `band`/`sum`/`max` combination, or a schedule of the distance below $\theta$ (`linear_ramp`) |
| multi-label: each label independently | `(N, C)` logits with per-label thresholds and boosts, and masks for missing labels |

With `mode="prob"`, one threshold and a boost below the floor, the loss equals Eq. 1
exactly; a test checks this.

## When it helps

In our evaluation (see [Results](../results.md)):

- **Where giving positives more weight helps** (few training positives, small models, small
  datasets): gains of about 1-3 points of sensitivity at high specificity, which a tuned
  weighted BCE matched.
- **Noisy positive labels, look-alike negatives that become more common after training, large
  networks trained for long:** no gain, or a loss.
- **Fine-tuning a model that is still learning:** it *hurts*. Boosting the hard positives
  before the model has learned the easy structure costs more than it gains. Train with BCE
  first, or use `warmup_steps`.
- **Very small reference classes** (around 100 frauds or malignant cases in the test set):
  sensitivity at 98-99% specificity is decided by one or two samples, and no method can be
  told apart from BCE.

See [Results](../results.md) for the comparison with weighted BCE, focal loss and LibAUC's
partial-AUC loss.
