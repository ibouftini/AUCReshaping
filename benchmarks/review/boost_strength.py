"""How strong is a boost on real scores? From the independent review, using saved scores only.

The review found that the reading of Eq. 1 sets the real strength of a boost, and calculated this on
idealised distributions. This module measures it on saved validation scores, with no training:

    python -m benchmarks.review.boost_strength --out benchmarks/results/s13a/strength

"Gradient mass" is the summed per-sample gradient magnitude on label 1 relative to BCE. It shows how
strong a boost is, and says nothing about whether the result improves.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aucreshape import auc_reshaping_loss, boost_values
from benchmarks.review import reanalysis

PAPER_THRESHOLDS = (0.90, 0.92, 0.95, 0.98)
PAPER_BOOSTS = (0.4, 0.2, 0.1, 0.1)
CONSTANT = (0.2, 0.2, 0.2, 0.2)
CONFIGS: dict[str, dict[str, Any]] = {
    "paper boosts, band, r=0.1 (as run)": {
        "specs": PAPER_THRESHOLDS, "boosts": PAPER_BOOSTS, "combine": "band", "min_ratio": 0.1,
    },
    "paper boosts, band, r=0.25": {
        "specs": PAPER_THRESHOLDS, "boosts": PAPER_BOOSTS, "combine": "band", "min_ratio": 0.25,
    },
    "paper boosts, band, r=0.5": {
        "specs": PAPER_THRESHOLDS, "boosts": PAPER_BOOSTS, "combine": "band", "min_ratio": 0.5,
    },
    "paper boosts, sum, r=0.1": {
        "specs": PAPER_THRESHOLDS, "boosts": PAPER_BOOSTS, "combine": "sum", "min_ratio": 0.1,
    },
    "0.2 at four thresholds, band, r=0.1": {
        "specs": PAPER_THRESHOLDS, "boosts": CONSTANT, "combine": "band", "min_ratio": 0.1,
    },
    "0.2 at four thresholds, sum, r=0.1": {
        "specs": PAPER_THRESHOLDS, "boosts": CONSTANT, "combine": "sum", "min_ratio": 0.1,
    },
}  # fmt: skip
E4_TASKS = {"effusion": 2, "pneumothorax": 7, "any finding": 14}


def measure(
    scores: np.ndarray,
    labels: np.ndarray,
    specs: Sequence[float],
    boosts: Sequence[float],
    *,
    combine: str = "band",
    min_ratio: float = 0.1,
    seed: int = 0,
) -> dict[str, float]:
    """Measure one boost configuration, in `prob` mode, on one set of scores.

    Parameters
    ----------
    scores : ndarray, shape (N,)
        Logits of label 1.
    labels : ndarray, shape (N,)
        1 for the boosted class, 0 for the class that sets the thresholds.
    specs, boosts : sequence of K floats
        Target specificities, ascending, and their boosts.
    combine : {"band", "sum", "max"}, default "band"
        How boosts combine.
    min_ratio : float, default 0.1
        The floor of `prob` mode.
    seed : int, default 0
        Seed of the random-recipient comparison.

    Returns
    -------
    dict
        ``share_flagged``: share of label 1 below a threshold. ``share_undefined``: share with
        ``p <= b``, where Eq. 1 is undefined. ``mass_ratio``: gradient mass relative to BCE.
        ``capped_share_of_extra``: share of the extra gradient from samples at the floor.
        ``random_extra_share``: extra gradient when the same boosts go to random label-1 samples, as
        a share of the targeted arm's. ``random_overlap``: share of those recipients also flagged.
    """
    z = torch.from_numpy(scores.astype(np.float64))
    y = torch.from_numpy(labels.astype(np.float64))
    negative = scores[labels < 0.5].astype(np.float64)
    thresholds = torch.tensor([np.quantile(negative, s) for s in specs], dtype=torch.float64)
    boost = torch.tensor(list(boosts), dtype=torch.float64)

    grad_z = z.clone().requires_grad_(True)
    loss = auc_reshaping_loss(
        grad_z, y, thresholds, boost, mode="prob", combine=combine, reduction="sum",  # type: ignore[arg-type]
        min_ratio=min_ratio,
    )  # fmt: skip
    loss.backward()
    assert grad_z.grad is not None
    positive = y > 0.5
    ours = grad_z.grad[positive].abs()
    p = torch.sigmoid(z[positive])
    base = 1.0 - p
    b = boost_values(z, y, thresholds, boost, combine=combine)[positive].to(p.dtype)  # type: ignore[arg-type]

    def magnitude(recipient_boost: torch.Tensor) -> torch.Tensor:
        capped = torch.minimum(recipient_boost, (1.0 - min_ratio) * p)
        return base / (1.0 - capped / p)

    flagged = b > 0
    capped_samples = flagged & (b >= (1.0 - min_ratio) * p)
    extra = ours - base
    total_extra = float(extra.sum())
    order = torch.randperm(len(b), generator=torch.Generator().manual_seed(seed))
    moved = b[order]
    random_extra = float((magnitude(moved) - base).sum())
    recipients = moved > 0
    return {
        "share_flagged": float(flagged.double().mean()),
        "share_undefined": float((flagged & (p <= b)).double().mean()),
        "mass_ratio": float(ours.sum() / base.sum()),
        "capped_share_of_extra": float(extra[capped_samples].sum()) / total_extra
        if total_extra > 0
        else 0.0,
        "random_extra_share": random_extra / total_extra if total_extra > 0 else float("nan"),
        "random_overlap": float(flagged[recipients].double().mean())
        if bool(recipients.any())
        else float("nan"),
    }


def e4_rows(root: Path, epochs: Sequence[int] = (5, 10)) -> list[dict[str, Any]]:
    """Measure the first chest X-ray study's CE models on validation scores, paper's labels."""
    rows = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
    table = []
    for run in rows:
        if run["arm"] != "ce":
            continue
        with np.load(root / "scores" / f"{run['key']}.npz") as saved:
            for epoch in epochs:
                for task, col in E4_TASKS.items():
                    z = -saved[f"val_{epoch}"][:, col].astype(np.float32)
                    y = 1.0 - saved["y_val"][:, col].astype(np.float32)
                    for name, config in CONFIGS.items():
                        table.append(
                            {
                                "study": "E4, CE, validation",
                                "dataset": task,
                                "seed": run["seed"],
                                "epoch": epoch,
                                "config": name,
                                **measure(z, y, config["specs"], config["boosts"],
                                          combine=config["combine"], min_ratio=config["min_ratio"]),
                            }
                        )  # fmt: skip
    return table


def e1_rows(root: Path, bundle: Path, scenarios: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    """Measure the synthetic laboratory's CE models on validation scores, at the tuned grid."""
    index = json.loads((bundle / "index.json").read_text())
    labels = dict(np.load(bundle / "labels.npz"))
    meta = json.loads((bundle / "meta.json").read_text())
    table = []
    for name, tag in scenarios:
        run = index[name]["prob"][tag]["ce"]["0"]
        with np.load(root / run["folder"] / "scores" / f"{run['key']}.npz") as saved:
            z = saved[f"val_{run['val']}"].astype(np.float32)
        key = f"{name}|{tag}|0|val"
        y = reanalysis.unpack_labels(labels[key], meta[key]).astype(np.float32)
        table.extend(
            {
                "study": "E1, CE, validation",
                "dataset": f"{name} {tag}",
                "seed": 0,
                "epoch": run["val"],
                "config": f"{boost} at 0.98, r=0.1",
                **measure(z, y, (0.98,), (boost,)),
            }
            for boost in (0.05, 0.1, 0.2, 0.4)
        )
    return table


def credit_card_rows(root: Path) -> list[dict[str, Any]]:
    """Measure the credit-card CE models without momentum; the split has no validation set."""
    table = []
    paths = sorted(glob.glob(str(root / "*" / "results" / "momentum_0_*.npz")))
    for path in paths:
        with np.load(path) as saved:
            z = saved["test_200"].astype(np.float32)
            y = saved["y_normal"].astype(np.float32)
        specs = (0.90, 0.92, 0.95, 0.98, 0.99)
        table.extend(
            {
                "study": "credit card, CE no momentum, test (baseline only)",
                "dataset": Path(path).stem,
                "seed": int(Path(path).stem.rsplit("_", 1)[1]),
                "epoch": 200,
                "config": f"0.2 at five thresholds, {combine}, r=0.1",
                **measure(z, y, specs, (0.2,) * 5, combine=combine),
            }
            for combine in ("band", "sum")
        )
    return table


def main() -> None:
    """Write the measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = (("S-3", "0.0"), ("S-4", "0.1"), ("S-6", "mlp_32_20000"), ("S-7", "200_0.02"))
    rows = (
        e4_rows(args.results / "s12" / "e4")
        + e1_rows(args.results / "s12" / "e1", args.results / "s13a" / "bundle", scenarios)
        + credit_card_rows(args.results / "s13a" / "cc" / "output")
    )
    reanalysis.write_csv(rows, args.out / "boost_strength.csv")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
