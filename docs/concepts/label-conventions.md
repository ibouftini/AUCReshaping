# Label conventions

!!! tip "The rule"
    **Label 1 is the class whose sensitivity you want to raise. Label 0 is the class whose
    specificity you hold fixed.**

The thresholds come from the **label-0 scores** (a quantile at the target specificity); the
**label-1 samples below them** are boosted.

## Medical convention (disease = 1)

"Detect disease with at most 2% false alarms on healthy cases." Use the loss as is:

```python
criterion = AUCReshapingLoss(0.98, boost=0.2)
loss = criterion(logits, disease_labels)
```

## The paper's convention (normal = 1)

The paper calls the **normal** class positive and measures specificity on the abnormal one:
"clear as many normal chest X-rays as possible while missing at most 2% of abnormal ones" (a
triage setting), or "pass as many normal transactions as possible while flagging at least 99% of
fraud". If your labels are `abnormal = 1`, negate the logits and flip the labels:

```python
loss = criterion(-logits, 1 - labels)  # normal is now positive, abnormal is the reference
```

Everything the criterion reports then lives in the flipped space:

- a threshold $\theta$ on `-logits` is the threshold $-\theta$ on `logits`, with "below" and "above"
  swapped;
- `pos_weight` weights the normal class;
- metrics need the same flip: `metrics.sensitivity_at_specificity(-logits, 1 - labels, 0.99)`.

A property test checks that the flipped tracker thresholds the abnormal scores and boosts
exactly the normal samples a fraud-catching threshold would flag.
