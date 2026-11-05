"""Tables of the fine-tuning lab, computed from its result rows (exploratory).

python -m benchmarks.review.report_lab --results DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from benchmarks.review.report_s13a import CLAIMS, current_wording

Row = dict[str, Any]
CONTRASTS = ("ce", "weighted_bce", "matched_random", "same_boost_random", "matched_above")


def load_levels(root: Path) -> list[Row]:
    """Merge every kernel's ``levels.jsonl``; failed levels stay in the list."""
    rows = [
        json.loads(line)
        for path in sorted(root.rglob("levels.jsonl"))
        for line in path.read_text().splitlines()
    ]
    for row in rows:
        for contrast in row.get("contrasts", ()):
            contrast["claim"] = current_wording(contrast["claim"])
    return rows


def failed(rows: Iterable[Row]) -> list[Row]:
    """Return the levels that did not finish."""
    return [row for row in rows if row.get("status") != "ok"]


def unique_levels(rows: Iterable[Row]) -> list[Row]:
    """Keep one finished row per experiment: some scenarios share a level, which counts once."""
    seen: dict[tuple[Any, ...], Row] = {}
    for row in rows:
        if row.get("status") == "ok":
            key = (row["regime"], *(round(v, 9) for v in row["arms"]["ce"]))
            seen.setdefault(key, row)
    return list(seen.values())


def _regime(rows: Iterable[Row], regime: str) -> list[Row]:
    return [row for row in unique_levels(rows) if row["regime"] == regime]


def claim_counts(rows: Iterable[Row], regime: str) -> dict[str, Counter[str]]:
    """Count the permitted wordings of AUCReshaping minus each arm, over unique levels."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _regime(rows, regime):
        for contrast in row["contrasts"]:
            counts[contrast["minus"]][contrast["claim"]] += 1
    return dict(counts)


def median_gain_over_ce(rows: Iterable[Row], regime: str) -> dict[str, float]:
    """Return each arm's median, over unique levels, of its mean sensitivity minus CE's."""
    gains: dict[str, list[float]] = defaultdict(list)
    for row in _regime(rows, regime):
        ce = statistics.fmean(row["arms"]["ce"])
        for arm, values in row["arms"].items():
            if arm != "ce":
                gains[arm].append(statistics.fmean(values) - ce)
    return {arm: statistics.median(values) for arm, values in gains.items()}


def edge_counts(
    rows: Iterable[Row], regime: str, arm: str, grid: Sequence[dict[str, Any]]
) -> tuple[int, int]:
    """Return how many unique levels chose the first and the last setting of a grid."""
    picks = [row["chosen"].get(arm) for row in _regime(rows, regime)]
    return picks.count(grid[0]), picks.count(grid[-1])


def _claim_table(rows: Sequence[Row], regime: str) -> list[str]:
    counts = claim_counts(rows, regime)
    lines = [
        "| AUCReshaping minus | " + " | ".join(CLAIMS) + " |",
        "|---|" + "---|" * len(CLAIMS),
    ]
    for arm in CONTRASTS:
        if arm in counts:
            cells = " | ".join(str(counts[arm][claim]) for claim in CLAIMS)
            lines.append(f"| {arm} | {cells} |")
    return lines


def _level_table(rows: Sequence[Row], regime: str) -> list[str]:
    lines = [
        "| Scenario | Level | CE | AR | Minus CE | Minus weighted BCE | Minus matched random |"
        " Chosen | Strength |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(_regime(rows, regime), key=lambda r: (r["name"], r["tag"])):
        by_arm = {c["minus"]: c for c in row["contrasts"]}
        cells = [
            f"{by_arm[arm]['delta']:+.2f} [{by_arm[arm]['lo']:+.2f}, {by_arm[arm]['hi']:+.2f}]"
            if arm in by_arm
            else ""
            for arm in ("ce", "weighted_bce", "matched_random")
        ]
        chosen = row["chosen"].get("aucreshape", {})
        setting = f"{chosen.get('combine', '')} {chosen.get('min_ratio', '')}".strip()
        lines.append(
            f"| {row['name']} | {row['tag']} | {statistics.fmean(row['arms']['ce']):.1f} "
            f"| {statistics.fmean(row['arms']['aucreshape']):.1f} | {cells[0]} | {cells[1]} "
            f"| {cells[2]} | {setting} | {row['diagnostics']['strength']['mass_ratio']:.1f}x |"
        )
    return lines


def markdown(rows: Sequence[Row]) -> str:
    """Return the lab's tables as Markdown; every number is exploratory."""
    lines = [f"Levels run: {len(rows)}. Failed: {len(failed(rows))}.", ""]
    for regime in ("rare", "paper"):
        kept = _regime(rows, regime)
        lines += [f"## Regime: {regime}, {len(kept)} unique levels", ""]
        lines += [*_claim_table(rows, regime), ""]
        medians = median_gain_over_ce(rows, regime)
        lines += [
            "Median gain over CE, in points: "
            + ", ".join(f"{arm} {value:+.2f}" for arm, value in sorted(medians.items())),
            "",
        ]
        lines += [*_level_table(rows, regime), ""]
    return "\n".join(lines)


def main() -> None:
    """Write the lab's tables to a file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown(load_levels(args.results)))


if __name__ == "__main__":
    main()
