# Distributed training (DDP)

Nothing changes. The tracker all-gathers the negatives' logits in `update`, so all ranks hold identical thresholds, and capacities count samples globally. Every rank must call the loss in training mode the same number of times; run validation under `torch.no_grad()` on all ranks. Launch with `torchrun --nproc_per_node=2 examples/ddp_torchrun.py --fast`.

```python
--8 < --"examples/ddp_torchrun.py"
```

Run it: `python examples/ddp_torchrun.py --fast` (CI runs every example).
