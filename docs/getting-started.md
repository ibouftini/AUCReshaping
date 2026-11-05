# Getting started

## Install

```bash
pip install "aucreshape @ git+https://github.com/ibouftini/AUCReshaping"              # needs only torch >= 2.1, Python >= 3.10
pip install "aucreshape[lightning] @ git+https://github.com/ibouftini/AUCReshaping"   # optional Lightning callback
```

## Ten lines

```python
import torch
from aucreshape import AUCReshapingLoss, metrics

x = torch.randn(4096, 10)
y = (x[:, 0] + x[:, 1] ** 2 + torch.randn(4096) > 2.5).float()  # imbalanced labels
model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
criterion = AUCReshapingLoss(0.95, boost=0.2, warmup_steps=150)  # was BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(200):
    loss = criterion(model(x).squeeze(-1), y)  # label 1 = the class to detect
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(metrics.sensitivity_at_specificity(model(x).squeeze(-1).detach(), y, 0.95))
```

## The recipe that works

1. **Train with BCE** until validation stops improving.
2. **Fine-tune with `AUCReshapingLoss`** for a few epochs at a lower learning rate, with
   `target_specificity` set to your operating point.
3. **Select checkpoints on** `metrics.sensitivity_at_specificity(val_logits, val_targets, s)`,
   not on the loss. Outside training the criterion returns plain BCE, so its value says
   nothing about the operating point.
4. **Pick the threshold you deploy with** on validation data:
   `metrics.threshold_at_specificity(val_logits, val_targets, s)`.

## Pitfalls

!!! warning "`model.eval()` does not switch the criterion"
    The criterion is its own module. Run validation under `torch.no_grad()` (or call
    `criterion.eval()` and later `criterion.train()`). Otherwise validation batches feed the
    threshold tracker and you get the boosted loss. The criterion warns when a training-mode
    loss is discarded without `backward()`.

- **Validation BCE is expected to rise** while sensitivity at the operating point improves.
  Do not early-stop or schedule on validation loss.
- **A script that trains from scratch:** keep its BCE epochs as warm-up
  (`warmup_steps = bce_epochs * len(train_loader)`, counted in micro-batches), lower the
  learning rate to 0.1 to 0.3 times when warm-up ends, fine-tune for 1 to 3 epochs and select the epoch
  on validation sensitivity. Gains often peak in the first fine-tuning epoch.
- If validation AUC is near 0.5, fix the model first.
- `criterion.last_stats` describes the last batch only.
- The default `window` holds the last `max(4096, 2 * min_negatives)` negatives. If an epoch has
  far fewer negatives than that, the threshold mixes several epochs' model snapshots; a
  `window_size` of about one epoch of negatives (but at least `min_negatives`) tracks a
  quickly changing model better.

## What happens inside

- On each training call, the loss is computed with the thresholds estimated *before* this
  batch. Then the batch's negatives are added to the threshold tracker.
- The tracker needs about `10 / (1 - s)` negatives before it starts boosting (200 at 0.95,
  500 at 0.98). Until then the loss is plain BCE. It warns if it never gets there.
- `criterion.last_stats` tells you what happened: the fraction of positives boosted, the
  thresholds, and whether each label is ready.

Next: [the method](concepts/method.md), or a [how-to guide](how-to/plain-pytorch.md) for your
framework.
