"""Hugging Face Trainer recipe: override compute_loss; keep the criterion in checkpoints.

A tiny MLP stands in for a transformer so the example needs no download. The criterion lives
outside the model, so a TrainerCallback saves its state next to each checkpoint and calls
end_epoch(); load it back yourself before resuming.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from _common import make_data, make_model, parse_args
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from aucreshape import AUCReshapingLoss

CRITERION_FILE = "aucreshape_criterion.pt"


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = make_model()

    def forward(self, features, labels=None):
        return SequenceClassifierOutput(logits=self.net(features))


class Rows(torch.utils.data.Dataset):
    def __init__(self, x, y) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return {"features": self.x[i], "labels": self.y[i]}


class ReshapingTrainer(Trainer):
    def __init__(self, *args, criterion: AUCReshapingLoss, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = criterion

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.criterion(outputs.logits.squeeze(-1), labels)
        return (loss, outputs) if return_outputs else loss


class CriterionCallback(TrainerCallback):
    """Save the criterion with every checkpoint and finalise its epochs."""

    def __init__(self, criterion: AUCReshapingLoss) -> None:
        self.criterion = criterion

    def on_save(self, args, state, control, **kwargs):
        folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        torch.save(self.criterion.state_dict(), folder / CRITERION_FILE)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.criterion.end_epoch()


def build(output_dir: str, epochs: int, x, y, x_val, y_val):
    torch.manual_seed(0)
    criterion = AUCReshapingLoss(0.95, boost=0.2)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=1024,
        learning_rate=1e-3,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="no",
        report_to=[],
        use_cpu=True,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = ReshapingTrainer(
        model=TinyClassifier(),
        args=args,
        train_dataset=Rows(x, y),
        eval_dataset=Rows(x_val, y_val),
        criterion=criterion,
        callbacks=[CriterionCallback(criterion)],
    )
    return trainer, criterion


def main() -> None:
    cli = parse_args(__doc__)
    n = 3_000 if cli.fast else 30_000
    x, y = make_data(n)
    x_val, y_val = make_data(n // 2, seed=1)
    with tempfile.TemporaryDirectory() as out:
        trainer, criterion = build(out, 2, x, y, x_val, y_val)
        trainer.train()
        print("eval (plain BCE):", trainer.evaluate()["eval_loss"])
        first = sorted(Path(out).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[0]

        resumed, resumed_criterion = build(out, 2, x, y, x_val, y_val)
        resumed_criterion.load_state_dict(torch.load(first / CRITERION_FILE))
        resumed.train(resume_from_checkpoint=str(first))
        torch.testing.assert_close(resumed_criterion.thresholds, criterion.thresholds)
        print(
            f"resumed from {first.name}: threshold {resumed_criterion.thresholds.item():.4f} "
            f"(uninterrupted {criterion.thresholds.item():.4f})"
        )


if __name__ == "__main__":
    main()
