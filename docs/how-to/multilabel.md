# Multi-label and missing labels

`num_labels=C` gives every label its own thresholds. Pass `mask=~torch.isnan(targets)`: masked elements contribute exactly 0 loss and 0 gradient and never reach the tracker, even when their targets are NaN.

```python
--8 < --"examples/multilabel_masked.py"
```

Run it: `python examples/multilabel_masked.py --fast` (CI runs every example).
