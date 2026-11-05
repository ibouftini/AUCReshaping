"""Kaggle kernel plans (benchmarks/kaggle/make_kernels.py writes them).

- ``s12a``: the smoke runs. ``aucreshape-s12-nih256`` (CPU) resizes NIH ChestX-ray14 to
  256 px once; later kernels attach its output. ``aucreshape-s12-throughput`` (2 x T4)
  measures one epoch of the SwAV ResNet-50 end-to-end protocol at 128 and 224 px (the
  MedMNIST+ sizes; 256 px costs about 1.3 times 224), one resolution per GPU, to fix the E4
  budget for 200-epoch runs and check how a 2 x T4 session counts against the weekly quota.
- ``s12e``: E4, end-to-end chest X-ray. SwAV ResNet-50 on the
  ``nih256`` output at 128 px, 200 epochs, one run per T4 (~6.5 h), two runs per kernel. Four
  arms (CE, the paper's AUCReshaping, class-weighted focal loss, the random-boost control) x
  seeds 0-2. The first two kernels hold one seed of every arm, so they are pushed first.
"""

from __future__ import annotations

import os

from benchmarks.kaggle.make_kernels import Kernel

NIH = "nih-chest-xrays/data"
NIH256 = f"{os.environ.get('KAGGLE_USER', 'KAGGLE_USER')}/aucreshape-s12-nih256"  # s12a output
E4_PAIRS = (("ce", "ar_paper"), ("focal", "random_boost"))


def _e4(arm: str, seed: int) -> list[str]:
    return ["benchmarks.replication", "nih", "--size", "128", "--epochs", "200",
            "--arms", arm, "--seeds", str(seed), "--data", "/kaggle/input",
            "--out", f"/kaggle/working/results/e4/{arm}_s{seed}"]  # fmt: skip


PLANS: dict[str, list[Kernel]] = {
    "s12a": [
        Kernel(
            "aucreshape-s12-nih256",
            jobs=[["benchmarks.review.nih", "/kaggle/input/data", "/kaggle/working/nih256"]],
            dataset_sources=[NIH],
        ),
        Kernel(
            "aucreshape-s12-throughput",
            jobs=[
                [
                    "benchmarks.replication",
                    "chestmnist",
                    "--size",
                    str(size),
                    "--arch",
                    "swav_resnet50",
                    "--epochs",
                    "1",
                    "--arms",
                    "ce",
                    "--seeds",
                    "0",
                    "--out",
                    f"/kaggle/working/results/throughput{size}",
                ]
                for size in (128, 224)
            ],
            gpu=True,
            workers=2,
        ),
    ],
    "s12e": [
        Kernel(
            f"aucreshape-s12e-e4-{2 * seed + half + 1}",
            jobs=[_e4(arm, seed) for arm in pair],
            gpu=True,
            workers=2,
            kernel_sources=[NIH256],
        )
        for seed in range(3)
        for half, pair in enumerate(E4_PAIRS)
    ],
}
