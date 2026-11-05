# FAQ

**Should I use AUCReshaping?**
Try a tuned `pos_weight` first. In our evaluation AUCReshaping helped only where up-weighting
positives helps in general (few positives, small models), and there a tuned weighted BCE did
as well or better. Avoid it with noisy positive labels, with look-alike negatives that may
become more common after deployment, and for large networks fine-tuned end to end for many
epochs. See [Results](results.md).

**My validation loss did not change after switching to `AUCReshapingLoss`.**
That is expected: outside training (`criterion.eval()` or `torch.no_grad()`) the loss is
plain BCE, so it stays comparable with your baseline. Select checkpoints on
`metrics.sensitivity_at_specificity`.

**Nothing is boosted (`last_stats["boosted_fraction"]` stays 0).**
The tracker is not ready yet: it needs about `10 / (1 - s)` negatives (`criterion.tracker.ready`,
`criterion.tracker.support`). With rare negatives, use `ThresholdTracker.from_scores` on a
validation pass or `strategy="epoch"`. Also check that `warmup_steps` has passed.

**Which class is "positive"?**
The one whose sensitivity you want to raise. See [Label conventions](concepts/label-conventions.md).

**Does it work with gradient accumulation / several heads?**
Yes. The tracker is updated on every training call, i.e. per micro-batch. Use one criterion
per head; calling the same criterion twice per step feeds both batches to its tracker.

**Does it work with AMP / `torch.compile` / DDP / FSDP?**
AMP, compile and DDP are tested. FSDP works as long as all ranks are in the default process
group; there is no `process_group` argument.

**Can I use it for multi-class (softmax) models?**
Not directly. To reshape one class, pass that class's one-vs-rest logit
`logits[:, k] - torch.logsumexp(other_logits, dim=1)` as a binary logit.

**How does it compare with LibAUC, focal loss or weighted BCE?**
LibAUC optimises (partial) AUC with pairwise surrogate losses; focal loss and weighted BCE
re-weight samples globally. AUCReshaping keeps BCE and re-weights only the positives
misclassified at your operating point. The [Results](results.md) page compares them under
the same budget.

**Is this the authors' code?**
No. It is an independent, clean-room implementation from the paper, not affiliated with the
authors or Siemens Healthineers.
