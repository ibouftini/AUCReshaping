# Thresholds from validation

The paper picks thresholds on validation data. `ThresholdTracker.from_scores` builds a `fixed` tracker at the exact validation operating point; `calibrate` refreshes it, e.g. every epoch.

```python
--8 < --"examples/thresholds_from_validation.py"
```

Run it: `python examples/thresholds_from_validation.py --fast` (CI runs every example).
