"""PyTorch Lightning integration (needs the ``[lightning]`` extra).

Examples
--------
>>> from aucreshape.integrations.lightning import AUCReshapingCallback  # doctest: +SKIP
>>> trainer = lightning.Trainer(callbacks=[AUCReshapingCallback()])  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

try:
    import lightning.pytorch as pl
except ImportError as error:  # pragma: no cover - exercised by a test that hides lightning
    raise ImportError(
        "aucreshape.integrations.lightning needs PyTorch Lightning. Install it with "
        '`pip install "aucreshape[lightning]"`.'
    ) from error

from aucreshape.loss import AUCReshapingLoss

__all__ = ["AUCReshapingCallback"]


class AUCReshapingCallback(pl.Callback):
    """Call ``end_epoch()`` on every ``AUCReshapingLoss`` of the module and log its stats.

    The criterion must be an attribute (submodule) of the ``LightningModule``, which also
    puts its state in Lightning checkpoints.

    Parameters
    ----------
    log_stats : bool, default True
        Log, per criterion and training batch: ``boosted_fraction``, ``ready_fraction`` and
        ``mean_boost`` (means over labels) and, once a label is ready, ``threshold`` (the
        operating-point threshold averaged over ready labels). Keys are
        ``{prefix}/{module_name}/{stat}``.
    prefix : str, default "aucreshape"
        Prefix of the logged keys.
    """

    def __init__(self, *, log_stats: bool = True, prefix: str = "aucreshape") -> None:
        super().__init__()
        self.log_stats = log_stats
        self.prefix = prefix

    @staticmethod
    def _criteria(pl_module: pl.LightningModule) -> list[tuple[str, AUCReshapingLoss]]:
        return [
            (name or "criterion", module)
            for name, module in pl_module.named_modules()
            if isinstance(module, AUCReshapingLoss)
        ]

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,  # noqa: ANN401 - Lightning's hook signature
        batch: Any,  # noqa: ANN401
        batch_idx: int,
    ) -> None:
        """Log the criteria's ``last_stats``."""
        if not self.log_stats:
            return
        for name, criterion in self._criteria(pl_module):
            stats = criterion.last_stats
            key = f"{self.prefix}/{name}"
            ready = stats["ready"].float()
            pl_module.log(f"{key}/boosted_fraction", stats["boosted_fraction"].float().mean())
            pl_module.log(f"{key}/ready_fraction", ready.mean())
            pl_module.log(f"{key}/mean_boost", stats["mean_boost"].float().mean())
            if bool(ready.any()):
                operating = stats["thresholds"][:, -1].float()
                value = (operating * ready).sum() / ready.sum()
                pl_module.log(f"{key}/threshold", value)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Finalise ``strategy="epoch"`` trackers and check readiness."""
        for _, criterion in self._criteria(pl_module):
            criterion.end_epoch()
