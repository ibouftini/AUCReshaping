"""Tables of the audit, generated from the result files (exploratory).

    python -m benchmarks.review.report_s13a --results benchmarks/results/s13a --out FILE

Reads the synthetic contrasts collected from Kaggle, the local validation-threshold and noise-floor
tables, the older studies' contrasts and the credit-card summaries. Writes one Markdown file.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from benchmarks.review.stats import T975

ARMS = ("random_boost", "boost_all", "placebo", "heldout", "oracle", "weighted_bce")
NAMES = {
    "random_boost": "same boosts, random positives",
    "boost_all": "boost every positive",
    "placebo": "boost detected positives",
    "heldout": "thresholds from validation",
    "oracle": "thresholds from the true distribution",
    "weighted_bce": "weighted BCE",
}
CLAIMS = (
    "gain of at least about 1 point",
    "gain, size uncertain",
    "no difference larger than 1 point",
    "inconclusive",
    "loss, size uncertain",
    "loss of at least about 1 point",
)


# Results saved before the wording changed keep their old text on disk.
_OLD_WORDING = {
    "small gain, size uncertain": "gain, size uncertain",
    "small loss, size uncertain": "loss, size uncertain",
}


def current_wording(claim: str) -> str:
    """Return a saved wording as it reads now."""
    return _OLD_WORDING.get(claim, claim)


def interval(delta: float, lo: float, hi: float) -> str:
    """Format a difference and its interval, in points."""
    return f"{delta:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def load_contrasts(root: Path) -> list[dict[str, Any]]:
    """Merge every kernel's ``contrasts.jsonl``; failed contrasts stay in the list."""
    rows = [
        json.loads(line)
        for path in sorted(root.rglob("contrasts.jsonl"))
        for line in path.read_text().splitlines()
    ]
    for row in rows:
        if "claim" in row:
            row["claim"] = current_wording(row["claim"])
    return rows


def claim_counts(rows: Iterable[dict[str, Any]]) -> Counter[tuple[str, str, str]]:
    """Count the permitted wordings per setting and arm."""
    return Counter(
        (row["setting"], row["arm"], row["claim"]) for row in rows if row.get("status") == "ok"
    )


def unique_cells(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per experiment: some scenarios share a level, which must not count twice."""
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row["setting"], row["arm"], *(round(v, 9) for v in row.get("per_seed", ())))
        seen.setdefault(key, row)
    return list(seen.values())


def seeds_needed(sd: float, half_width: float = 1.0) -> int:
    """Smallest number of paired seeds whose t interval has at most this half-width.

    It covers seed variance only; test sampling widens the interval further. Beyond 31 seeds the
    quantile is taken as 2.0.
    """
    n = 2
    while True:
        t = T975[n - 2] if n - 2 < len(T975) else 2.0
        if t * sd / n**0.5 <= half_width:
            return n
        n += 1


def scenario_table(
    rows: Sequence[dict[str, Any]], name: str, setting: str, arms: Sequence[str]
) -> list[str]:
    """One Markdown row per level: AUCReshaping minus each arm, in points."""
    chosen = [r for r in rows if r["name"] == name and r["setting"] == setting]
    tags = list(dict.fromkeys(r["tag"] for r in chosen))
    cells = {(r["tag"], r["arm"]): r for r in chosen}
    lines = ["| level | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for tag in tags:
        out = []
        for arm in arms:
            row = cells.get((tag, arm))
            if row is None:
                out.append("n/a")
            elif row.get("status") != "ok":
                out.append("failed")
            else:
                out.append(interval(row["delta"], row["lo"], row["hi"]))
        lines.append(f"| {tag} | " + " | ".join(out) + " |")
    return lines


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _contrast_section(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Synthetic laboratory: direct contrasts", ""]
    failed = [r for r in rows if r.get("status") != "ok"]
    unique = unique_cells(r for r in rows if r.get("status") == "ok")
    lines += [
        f"{len(rows)} contrasts, {len(failed)} failed, {len(unique)} unique: S-3 at 0.02, S-5 at",
        "0.0 and S-8 at 0.02 are one experiment, so the counts below use each once. Intervals",
        "carry no correction for multiple comparisons. Each cell is AUCReshaping minus the arm,",
        "in points.",
        "",
    ]
    counts = claim_counts(unique)
    for setting in ("prob", "weight"):
        lines += [f"### Permitted wording, `{setting}`", ""]
        lines += ["| AUCReshaping minus | " + " | ".join(CLAIMS) + " |"]
        lines += ["|---|" + "---|" * len(CLAIMS)]
        for arm in ARMS:
            values = [str(counts.get((setting, arm, claim), 0)) for claim in CLAIMS]
            lines += [f"| {NAMES[arm]} | " + " | ".join(values) + " |"]
        lines += [""]
    for name in sorted({r["name"] for r in rows}):
        for setting in ("prob", "weight"):
            if any(r["name"] == name and r["setting"] == setting for r in rows):
                lines += [f"### {name}, `{setting}`", ""]
                lines += scenario_table(rows, name, setting, ARMS)
                lines += [""]
    return lines


def generic_table(
    rows: Sequence[dict[str, str]], fields: Sequence[str], digits: int = 2
) -> list[str]:
    """Return a Markdown table of CSV rows, rounding numbers and per-seed lists."""

    def cell(value: str) -> str:
        if value.isdigit():
            return value
        try:
            return f"{float(value):.{digits}f}"
        except ValueError:
            pass
        if value.startswith("["):
            return " / ".join(f"{float(v):.{digits}f}" for v in json.loads(value))
        return value

    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    lines += ["| " + " | ".join(cell(r.get(f, "")) for f in fields) + " |" for r in rows]
    return lines


def seed_spread(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired seed SD per setting and arm, and the seeds its median would need."""
    groups: dict[tuple[str, str], list[float]] = {}
    for row in unique_cells(r for r in rows if r.get("status") == "ok"):
        groups.setdefault((row["setting"], row["arm"]), []).append(row["paired_sd"])
    table = []
    for (setting, arm), values in sorted(groups.items()):
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        )
        table.append(
            {
                "setting": setting,
                "arm": arm,
                "cells": len(values),
                "median_sd": median,
                "max_sd": ordered[-1],
                "seeds_for_median": seeds_needed(median),
            }
        )
    return table


def build(results: Path) -> list[str]:
    """Return the lines of the tables file."""
    lines = ["# Audit: generated tables", "", "Exploratory, except the pre-registered rule.", ""]
    contrasts = load_contrasts(results / "contrasts" / "output")
    if contrasts:
        lines += _contrast_section(contrasts)
    older = results / "older"
    lines += ["## Chest X-ray (E4): direct contrasts", ""]
    lines += ["Interval: test sampling only, by image; 2 seeds, so no claim is attached.", ""]
    lines += generic_table(
        _csv(older / "e4_contrasts.csv"),
        ("task", "convention", "rule", "arm", "ar_sens", "control_sens", "delta", "lo", "hi"),
    )
    lines += ["", "## Chest X-ray (E4): selection on effusion alone", ""]
    lines += generic_table(
        _csv(older / "e4_task_selected.csv"), ("arm", "per_seed_epoch", "per_seed", "mean")
    )
    lines += ["", "## Chest X-ray (E4): validation threshold, effusion", ""]
    rows = [r for r in _csv(older / "e4_validation_threshold.csv") if r["task"] == "effusion"]
    lines += generic_table(
        rows, ("arm", "convention", "epoch", "sens_at_val_threshold", "realised_specificity")
    )
    lines += ["", "## Early benchmark: direct contrasts", ""]
    lines += generic_table(
        _csv(older / "s09_contrasts.csv"),
        (
            "dataset",
            "target",
            "setting",
            "arm",
            "ar_sens",
            "control_sens",
            "delta",
            "lo",
            "hi",
            "paired_sd",
            "claim",
        ),
    )
    lines += ["", "## First credit-card run: direct contrasts", ""]
    lines += generic_table(
        _csv(older / "s11_contrasts.csv"),
        ("rule", "ar_sens", "control_sens", "delta", "lo", "hi", "paired_sd", "claim"),
    )
    lines += ["", "## Noise floor: differences between baseline repeats", ""]
    floor = _csv(older / "noise_floor_older.csv")
    lines += generic_table(floor, ("study", "dataset", "pairs", "median_abs", "p95_abs", "max_abs"))
    lines += [""]
    lines += generic_table(
        _csv(results / "e1" / "noise_floor_e1.csv"),
        ("name", "tag", "pairs", "median_abs", "p95_abs", "max_abs"),
    )
    cc = results / "cc" / "output" / "summary_epoch200.csv"
    if cc.exists():
        lines += ["", "## Credit card: reconciling the baseline", ""]
        lines += generic_table(
            _csv(cc),
            (
                "config",
                "split_seed",
                "runs",
                "failed",
                "auc_mean",
                "sens@0.98_mean",
                "sens@0.98_sd",
                "sens@0.99_mean",
                "sens@0.99_sd",
                "sens@0.9999_mean",
            ),
            digits=3,
        )
    if contrasts:
        lines += ["", "## Paired seed SD on the synthetic laboratory", ""]
        spread = [{k: str(v) for k, v in row.items()} for row in seed_spread(contrasts)]
        lines += generic_table(
            spread, ("setting", "arm", "cells", "median_sd", "max_sd", "seeds_for_median")
        )
    return lines


def main() -> None:
    """Write the tables file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results/s13a"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text("\n".join(build(args.results)) + "\n")


if __name__ == "__main__":
    main()
