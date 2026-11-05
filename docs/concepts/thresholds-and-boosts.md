# Thresholds and boosts

## Target specificities

`target_specificity` is the operating point you deploy at, e.g. `0.95` or `0.98`. You can
give several, listed from the lowest to the highest specificity:

```python
AUCReshapingLoss([0.90, 0.95, 0.98], boost=[0.4, 0.2, 0.1])
```

The thresholds cut a positive's logit axis into bands:

| logit of a positive | below $\theta_{0.90}$ | from $\theta_{0.90}$ to $\theta_{0.95}$ | from $\theta_{0.95}$ to $\theta_{0.98}$ | above $\theta_{0.98}$ |
|---|---|---|---|---|
| boost (`band`) | `boost[0]` (far below the operating point) | `boost[1]` | `boost[2]` | none (at the operating point or above) |

| `combine` | boost of a positive below some thresholds |
|---|---|
| `"band"` (default) | the boost of its band (the lowest-specificity threshold it falls below) |
| `"sum"` | the sum of the boosts of all thresholds it falls below |
| `"max"` | the largest of those boosts |

Use `band` with boosts that **decrease** with specificity (the paper's
`[0.4, 0.2, 0.1, 0.1]`), and `sum` when every threshold has the **same** boost (the paper's
"0.2 at 0.90, 0.92, 0.95, 0.98"). Avoid boosting far below the target (e.g. 0.70 when
deploying at 0.98); the paper reports that it hurts.

## Boost magnitude and mode

| `mode` | the boost is | typical range |
|---|---|---|
| `"prob"` (default, the paper) | a probability offset: $-\log(p - b)$ | 0.05-0.4 |
| `"logit"` | a margin in logit units: $-\log \sigma(z - b)$ | 1-4 |
| `"weight"` | a loss weight: $(1 + b)\,\ell_{\mathrm{BCE}}$ | 0.5-4 |

Sweep 3-4 values on **validation sensitivity at the target specificity**. Our evaluation
used a grid of four values per mode.

A boost can also be a function of the distance $d = \theta - z$ below a single threshold, which is
the paper's "continuous" variant:

```python
from aucreshape import linear_ramp

AUCReshapingLoss(0.98, boost=linear_ramp(slope=0.2, max_boost=0.4))
```

## Threshold strategies

| `strategy` | estimate from | ready when | trade-off |
|---|---|---|---|
| `"window"` (default) | a FIFO buffer of the last `window_size` negatives | the buffer holds at least `min_negatives` | smooth and adaptive; best noise/lag trade-off in our tests |
| `"batch"` | the current batch's negatives | the batch has at least `min_negatives` | freshest; needs very large batches |
| `"ema"` | moving average of batch quantiles | at least `min_negatives` seen | cheap; biased with small batches |
| `"epoch"` | a uniform reservoir over the epoch, finalised by `end_epoch()` | after an epoch | stable; lags one epoch |
| `"fixed"` | `initial_thresholds`, `calibrate()` or `from_scores()` | always | full control, e.g. thresholds from validation |

`min_negatives` defaults to $\lceil 10/(1 - s_{\max}) \rceil$, where $s_{\max}$ is the highest target specificity: 200 at 0.95, 500 at 0.98, 1000
at 0.99. With fewer than about 10 negatives above the threshold, a quantile is noise. Until a
label is ready it gets no boost (plain BCE), and the tracker warns after 1000 updates and at
`end_epoch()` if it is still not ready.

The thresholds used for a batch were computed *before* that batch, so a batch never
influences its own boosts. Under DDP the negatives are all-gathered, so every rank has the
same thresholds.
