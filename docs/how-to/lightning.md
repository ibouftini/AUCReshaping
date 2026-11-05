# PyTorch Lightning

Make the criterion an attribute of the `LightningModule` (so it is a submodule and lands in Lightning checkpoints) and add `AUCReshapingCallback`, which calls `end_epoch()` and logs `boosted_fraction`, `ready_fraction`, `mean_boost` and the operating-point threshold. Install with `pip install "aucreshape[lightning]"`.

```python
--8 < --"examples/lightning_example.py"
```

Run it: `python examples/lightning_example.py --fast` (CI runs every example).
