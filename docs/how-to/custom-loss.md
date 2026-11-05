# Your own per-sample loss

`reshaping_weights` returns `1 + b` per element (detached). Multiply any per-sample loss, here a focal loss, and update a `ThresholdTracker` after using its thresholds.

```python
--8 < --"examples/custom_loss_weights.py"
```

Run it: `python examples/custom_loss_weights.py --fast` (CI runs every example).
