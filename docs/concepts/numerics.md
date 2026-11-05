# Numerics

## The floor in `prob` mode

Eq. 1 uses $-\log(p - b)$. When $p \le b$ this is undefined, and the paper's own boosts (up to
1.0) make that common. Three fixes were compared:

| fix | gradient for very hard positives |
|---|---|
| naive clamping, $\max(p - b, \varepsilon)$ | **zero**: the hardest positives stop learning |
| linear extension of $-\log$ below $\varepsilon$ | explodes (about 2000 times BCE) just past $p = b$, vanishes further away |
| **detached relative cap** (used) | between 1 and $1/r$ times BCE, never zero |

With $r$ = `min_ratio` and $\operatorname{sg}(\cdot)$ the stop-gradient (`.detach()`), the boost
is capped at

```math
\tilde b = \min\bigl(b,\ (1 - r)\,\operatorname{sg}(p)\bigr),
\qquad\text{so}\qquad p - \tilde b \ge r\,p .
```

Because the cap is a constant for autograd, the gradient with respect to the logit is
$-(1 - p)/(1 - \tilde b / p)$. Eq. 1 is reproduced exactly wherever the cap is inactive. The value is
computed in log space, so the loss and its gradient are finite for every finite float32
logit (a Hypothesis property test covers $\pm 3 \cdot 10^{38}$).

## dtype and devices

- float32 and float64 are computed in their own precision.
- float16 and bfloat16 logits (e.g. under `torch.autocast`) are computed in float32, and the
  loss is returned in float32, as `BCEWithLogitsLoss` does under autocast.
- Thresholds are stored in logit space and in float32 buffers.

## No host syncs

The forward pass and the tracker update never call `.item()` or `.cpu()` and never branch in
Python on tensor values. This is checked on CUDA with
`torch.cuda.set_sync_debug_mode("error")` for every strategy. Exceptions, documented in the
API: the readiness check at `end_epoch()` and at the 1000th update, `calibrate()`, the
metrics, and the all-gather collectives under DDP.

## Cost

The reshaped loss is a chain of elementwise ops, against BCE's single fused kernel. Eager
mode costs about 4 times BCE for the functional loss and 5 to 10 times for the module with its
tracker;
that is 0.3 to 3.6 ms per step for $N \le 4096$. `torch.compile` brings the functional loss to
about 1.7 to 2 times BCE (measured on a laptop CPU).
