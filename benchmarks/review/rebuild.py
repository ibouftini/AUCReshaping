"""Rebuild every table from saved results, without training.

    python -m benchmarks.review.rebuild [--results benchmarks/results] [--out DIR]

A step whose inputs are absent is skipped and says so: the public repository holds the summaries
but not the per-sample scores, so the steps that read scores need those downloaded first.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Step:
    """One rebuild: a name, the paths it reads, and what it does."""

    name: str
    needs: tuple[Path, ...]
    action: Callable[[], None]


def run(steps: Sequence[Step]) -> list[dict[str, str]]:
    """Run the steps in order; a missing input skips a step and a failure does not stop the rest."""
    report = []
    for step in steps:
        absent = [str(path) for path in step.needs if not path.exists()]
        if absent:
            report.append(
                {"step": step.name, "status": "skipped", "detail": "absent: " + ", ".join(absent)}
            )
            continue
        try:
            step.action()
        except Exception as error:  # every step is reported, then the rest still run
            report.append({"step": step.name, "status": "failed", "detail": repr(error)})
            continue
        report.append({"step": step.name, "status": "rebuilt", "detail": ""})
    return report


def exit_code(report: Sequence[dict[str, str]]) -> int:
    """Return 1 if a step failed; a skipped step is not a failure."""
    return int(any(entry["status"] == "failed" for entry in report))


def steps(results: Path, out: Path) -> list[Step]:
    """Return the real steps, from the lowest level up; none of them trains a model."""
    from benchmarks.review import (
        boost_strength,
        cc_reconcile,
        reanalysis,
        reanalysis_older,
        report_e4,
        report_s13a,
    )

    s13a = results / "s13a"
    e4 = results / "s12" / "e4"
    e1 = results / "s12" / "e1"
    bundle = s13a / "bundle"
    cc_output = s13a / "cc" / "output"
    cc_scores = cc_output / "kernel-0" / "results" / "paper_1500_1500.npz"

    def e4_summary() -> None:
        report_e4.write(report_e4.summary(e4), e4 / "summary.csv")

    def e1_local() -> None:
        tree = json.loads((bundle / "index.json").read_text())
        import numpy as np

        labels = dict(np.load(bundle / "labels.npz"))
        meta = json.loads((bundle / "meta.json").read_text())
        thresholds, floors = reanalysis.local_tables(e1, tree, labels, meta)
        reanalysis.write_csv(thresholds, s13a / "e1" / "validation_threshold.csv")
        reanalysis.write_csv(floors, s13a / "e1" / "noise_floor_e1.csv")

    def credit_card() -> None:
        rows = cc_reconcile.summarise(cc_output, 200)
        names = list(dict.fromkeys(name for row in rows for name in row))
        with (cc_output / "summary_epoch200.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, names)
            writer.writeheader()
            writer.writerows(rows)

    def older() -> None:
        argv = sys.argv
        sys.argv = ["reanalysis_older", "--out", str(s13a / "older"), "--results", str(results)]
        try:
            reanalysis_older.main()
        finally:
            sys.argv = argv

    def strength() -> None:
        scenarios = (("S-3", "0.0"), ("S-4", "0.1"), ("S-6", "mlp_32_20000"), ("S-7", "200_0.02"))
        rows = (
            boost_strength.e4_rows(e4)
            + boost_strength.e1_rows(e1, bundle, scenarios)
            + boost_strength.credit_card_rows(cc_output)
        )
        reanalysis.write_csv(rows, s13a / "strength" / "boost_strength.csv")

    def tables() -> None:
        (out / "S13a_tables.md").write_text("\n".join(report_s13a.build(s13a)) + "\n")

    return [
        Step("chest X-ray summary", (e4 / "scores",), e4_summary),
        Step(
            "synthetic validation threshold and noise floor",
            (bundle, e1 / "S-1" / "1.0" / "seed0" / "scores"),
            e1_local,
        ),
        Step("older studies, contrasts", (e4 / "scores", results / "tabular" / "raw"), older),
        Step("credit-card baseline", (cc_scores,), credit_card),
        Step("boost strength", (e4 / "scores", bundle, cc_scores), strength),
        Step("audit tables", (s13a / "older", s13a / "e1"), tables),
    ]


def main() -> None:
    """Run every step and print what happened to each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/reports"))
    args = parser.parse_args()
    report: list[dict[str, Any]] = list(run(steps(args.results, args.out)))
    for entry in report:
        print(f"{entry['status']:8s} {entry['step']} {entry['detail']}")
    sys.exit(exit_code(report))


if __name__ == "__main__":
    main()
