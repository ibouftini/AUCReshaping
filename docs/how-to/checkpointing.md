# Checkpoint and resume

All state (thresholds, buffers, the step counter) is in `criterion.state_dict()`. Resuming gives bit-identical training. Loading a state saved with a different configuration raises an error that names the difference.

```python
--8 < --"examples/checkpoint_resume.py"
```

Run it: `python examples/checkpoint_resume.py --fast` (CI runs every example).
