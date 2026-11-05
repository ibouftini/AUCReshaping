# Plain PyTorch

Pretrain with BCE, then fine-tune with `AUCReshapingLoss`; nothing else in the loop changes. `end_epoch()` finalises `strategy="epoch"` and checks readiness.

```python
--8 < --"examples/plain_pytorch.py"
```

Run it: `python examples/plain_pytorch.py --fast` (CI runs every example).
