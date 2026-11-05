# aucreshape

`aucreshape` is a PyTorch implementation of
[AUCReshaping](https://doi.org/10.1038/s41598-023-48482-x) (Bhat et al., *Scientific Reports*
13, 21097, 2023), and an evaluation of when the method helps. While you fine-tune a
classifier, AUCReshaping boosts the loss of the positives that are misclassified at a
high-specificity threshold, for example the 95th or 98th percentile of the negatives' scores,
with the aim of raising sensitivity at that operating point.

```python
from aucreshape import AUCReshapingLoss

criterion = AUCReshapingLoss(target_specificity=0.95, boost=0.2)  # was nn.BCEWithLogitsLoss()
loss = criterion(logits, targets)
```

It has the same call as `BCEWithLogitsLoss` (with `reduction`, `pos_weight` and `weight`),
and outside training it is plain BCE. It handles binary and multi-label `(N, C)` logits, masks
for missing labels, any model and data loader, CPU and CUDA, mixed precision,
`torch.compile`, DDP, checkpointing, Lightning and Hugging Face. The loss and its gradients
stay finite for every finite logit.

The library is finished and tested. It is not on PyPI; see [Getting started](getting-started.md)
to install it from GitHub. Its evaluation is being redone under a stricter protocol, published
before the new runs, because a review found gaps in the first one. The text below describes the
first evaluation, and may change.

**What the evaluation found.** AUCReshaping can raise sensitivity at high specificity by
1-3 points where giving the positive class more weight helps in general: few positives, small
training sets, small models. In those cases a tuned weighted BCE (`pos_weight`) does as well
or better, and boosting the same number of randomly chosen positives does as well as boosting
the missed ones. It does not help, and can lose several points, with mislabelled positives,
with negatives that resemble positives, when the data drift after training, and with large
networks trained for long. The figures, numbers and protocol are on the
[Results](results.md) page.

!!! tip "One rule to remember"
    **Label 1 is the class whose sensitivity you want to raise; label 0 is the class whose
    specificity you hold fixed.** See [Label conventions](concepts/label-conventions.md).

!!! warning "Fine-tuning, not training from scratch"
    AUCReshaping is a fine-tuning method. Train with BCE first, then switch the loss. On a
    model that is still learning the bulk of the data it can hurt; see
    [the method](concepts/method.md#when-it-helps).
