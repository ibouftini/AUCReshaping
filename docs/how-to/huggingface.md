# Hugging Face Trainer

Override `compute_loss`. The criterion lives outside the model, so the `TrainerCallback` saves its state next to each checkpoint and calls `end_epoch()`; load it back before `trainer.train(resume_from_checkpoint=...)`. Evaluation runs under `torch.no_grad()`, so `eval_loss` is plain BCE and the tracker is untouched.

```python
--8 < --"examples/hf_trainer_example.py"
```

Run it: `python examples/hf_trainer_example.py --fast` (CI runs every example).
