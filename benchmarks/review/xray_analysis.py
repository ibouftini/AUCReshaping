"""The confirmatory analysis of the chest X-ray study, fixed and tested before any test result.

    python -m benchmarks.review.xray_analysis integrity --results DIR
    python -m benchmarks.review.xray_analysis analyse --results DIR --codes FILE --out FILE
    python -m benchmarks.review.xray_analysis coverage --out FILE

``integrity`` reads coded runs and reports only whether they are complete. ``analyse`` needs the
file of arm codes, and is run once, after the code is frozen. Every run counts: a failed run is
shown, and its seed leaves only the contrasts that need it.

Scores are saved as disease logits and labels with disease as 1. They are read in the paper's
direction: images without the disease are the boosted class, and specificity is measured on disease.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch

from aucreshape import metrics
from benchmarks.review import analysis, reanalysis, stats
from benchmarks.review.report_s13a import CLAIMS as WORDINGS

TARGET = 0.98
RULES = ("val_sens", "val_loss")
PRIMARY_RULE = "val_sens"
AR_TUNED, AR_6X = "aucreshape", "aucreshape_6x"
# Name, ours, minus. P1 sits at 6x, because at the gentle chosen setting AUCReshaping and its
# control differ in only about 3% of the gradient, too little for a targeting test.
FAMILY = (
    ("P1", AR_6X, "matched_random_6x"),
    ("P2", AR_TUNED, "weighted_bce"),
    ("P3", AR_TUNED, "ce"),
)
SECONDARY = (("S1", AR_6X, "matched_above_6x"), ("S2", AR_6X, "weighted_bce"), ("S3", AR_6X, "ce"))
# A control that falls below CE would make a gain on P1 or S1 look like targeting.
CHECKS = (("C1", "matched_random_6x", "ce"), ("C2", "matched_above_6x", "ce"))
NOT_A_CLAIM = "sensitivity analysis, not a claim: "
_DIFFERENCES = {w for w in WORDINGS if "gain" in w or "loss" in w}
Entry = dict[str, Any]


def load_study(root: Path) -> list[Entry]:
    """Read every run's record and, if it finished, its saved scores under ``arrays``."""
    study = []
    for path in sorted(root.rglob("runs.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["status"] == "ok":
                with np.load(path.parent / f"{row['key']}.npz") as saved:
                    row["arrays"] = {name: saved[name] for name in saved.files}
            study.append(row)
    return study


def integrity(study: Sequence[Entry], expected: Sequence[str] | None = None) -> dict[str, Any]:
    """Report whether coded runs are complete, without reading anything that separates the arms.

    Parameters
    ----------
    study : sequence of dict
        From ``load_study``.
    expected : sequence of str, optional
        The keys of every planned run. A killed kernel leaves no record, so only this list can show
        that a run is missing.

    Returns
    -------
    dict
        Counts, codes and seeds; ``not_ok``, ``not_finite``, ``missing`` and ``duplicates``, as
        run keys; ``code_hashes`` and ``data_fingerprints``, which must hold one value each.
    """
    finished = [row for row in study if row["status"] == "ok"]
    keys = Counter(row["key"] for row in study)
    return {
        "runs": len(study),
        "codes": sorted({row["config"]["code"] for row in study}),
        "seeds": sorted({row["config"]["seed"] for row in study}),
        "not_ok": sorted(row["key"] for row in study if row["status"] != "ok"),
        "not_finite": sorted(
            row["key"]
            for row in finished
            if not all(np.isfinite(array).all() for array in row["arrays"].values())
        ),
        "missing": sorted(set(expected or ()) - set(keys)),
        "duplicates": sorted(key for key, count in keys.items() if count > 1),
        "code_hashes": sorted({str(row.get("code")) for row in study}),
        "data_fingerprints": sorted({json.dumps(row.get("data"), sort_keys=True) for row in study}),
    }


def sens_points(disease_logits: np.ndarray, disease: np.ndarray) -> float:
    """Return sensitivity on no-effusion images at 98% specificity on effusion, in points."""
    return _oriented_sens(-disease_logits, 1 - disease)


def _oriented_sens(scores: np.ndarray, labels: np.ndarray) -> float:
    return 100.0 * analysis.sens_at_spec(scores, labels, TARGET)


def _auc(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 1.0) -> float:
    value = metrics.partial_auc(
        torch.from_numpy(scores.astype(np.float64)),
        torch.from_numpy(labels.astype(np.float64)),
        max_fpr=max_fpr,
        standardized=max_fpr < 1.0,
    )
    return float(value)


def read_run(row: Entry, rule: str) -> dict[str, Any]:
    """Read one run at the epoch its validation data selected under a rule.

    Returns
    -------
    dict
        ``seed``, ``epoch``; ``sens_roc``, sensitivity read from the test ROC at 98%;
        ``sens_at_val_threshold`` and ``realised_specificity``, at the threshold fixed on
        validation; ``sens_no_finding``, on images labelled No Finding against disease, or None if
        not saved, all in points; ``auc`` and ``pauc``, up to a 2% false-alarm rate, as fractions.
    """
    epoch = int(row["selected"][rule])
    arrays = row["arrays"]
    test = -arrays[f"test_{epoch}"]
    val = -arrays[f"val_{epoch}"]
    y_test = 1 - arrays["y_test"].astype(np.float64)
    y_val = 1 - arrays["y_val"].astype(np.float64)
    sensitivity, specificity = reanalysis.validation_threshold(val, y_val, test, y_test, TARGET)
    no_finding = None
    if "no_finding_test" in arrays:
        keep = (y_test < 0.5) | (arrays["no_finding_test"] == 1)
        no_finding = _oriented_sens(test[keep], y_test[keep])
    return {
        "seed": int(row["config"]["seed"]),
        "epoch": epoch,
        "sens_roc": _oriented_sens(test, y_test),
        "sens_at_val_threshold": 100.0 * sensitivity,
        "realised_specificity": 100.0 * specificity,
        "sens_no_finding": no_finding,
        "auc": _auc(test, y_test),
        "pauc": _auc(test, y_test, 1 - TARGET),
    }


def family_claims(intervals: Sequence[tuple[float, float, float]]) -> list[str]:
    """Return the permitted wording of each contrast in the primary family.

    Parameters
    ----------
    intervals : sequence of (lo, hi, p)
        Each contrast's 95% interval, in points, and unadjusted two-sided p-value.

    Returns
    -------
    list of str
        The wording its interval permits. A gain or a loss must also survive Holm correction across
        the family, or it reads as inconclusive. "No difference" is not a discovery, so Holm leaves
        it as it is.
    """
    adjusted = stats.holm([p for _, _, p in intervals])
    claims = []
    for (lo, hi, _), p in zip(intervals, adjusted, strict=True):
        claim = reanalysis.claim_label(lo, hi)
        claims.append("inconclusive" if claim in _DIFFERENCES and p >= 0.05 else claim)
    return claims


def _above_cut(shifted: np.ndarray, labels: np.ndarray) -> float:
    return 100.0 * float((shifted[labels > 0.5] > 0).mean())


def _below_cut(shifted: np.ndarray, labels: np.ndarray) -> float:
    return 100.0 * float((shifted[labels < 0.5] <= 0).mean())


def _at_val_threshold(
    y: np.ndarray,
    patients: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int,
) -> dict[str, float]:
    """Compare the arms at thresholds fixed on validation; ``a`` and ``b`` are shifted by them."""
    both = stats.patient_bootstrap(y, patients, a, b, _above_cut, n_boot=n_boot)
    per_seed = [_above_cut(b[i], y) - _above_cut(a[i], y) for i in range(len(a))]
    lo, hi = stats.conservative_interval((both.lo, both.hi), stats.paired_t_interval(per_seed))
    specificity = [_below_cut(b[i], y) - _below_cut(a[i], y) for i in range(len(a))]
    return {
        "delta": both.point,
        "lo": lo,
        "hi": hi,
        "realised_specificity_delta": float(np.mean(specificity)),
    }


def _contrast(
    name: str,
    other: str,
    by_arm: dict[str, dict[int, Entry]],
    failed_by_arm: dict[str, set[int]],
    rule: str,
    y: np.ndarray,
    patients: np.ndarray,
    n_boot: int,
    own: str = "aucreshape",
) -> dict[str, Any]:
    ours, theirs = by_arm[own], by_arm[other]
    seeds = sorted(set(ours) & set(theirs))

    def stack(runs: dict[int, Entry]) -> np.ndarray:
        return np.stack(
            [-runs[s]["arrays"][f"test_{int(runs[s]['selected'][rule])}"] for s in seeds]
        )

    def cuts(runs: dict[int, Entry]) -> np.ndarray:
        values = []
        for s in seeds:
            arrays = runs[s]["arrays"]
            val = -arrays[f"val_{int(runs[s]['selected'][rule])}"]
            values.append(
                reanalysis.threshold_cut(val, 1 - arrays["y_val"].astype(np.float64), TARGET)
            )
        return np.array(values)[:, None]

    b, a = stack(ours), stack(theirs)
    both = stats.patient_bootstrap(y, patients, a, b, _oriented_sens, n_boot=n_boot)
    by_seeds = stats.patient_bootstrap(
        y, patients, a, b, _oriented_sens, n_boot=n_boot, resample="seeds"
    )
    by_patients = stats.patient_bootstrap(
        y, patients, a, b, _oriented_sens, n_boot=n_boot, resample="patients"
    )
    ours_sens = [_oriented_sens(b[i], y) for i in range(len(seeds))]
    theirs_sens = [_oriented_sens(a[i], y) for i in range(len(seeds))]
    per_seed = [x - z for x, z in zip(ours_sens, theirs_sens, strict=True)]
    t_lo, t_hi = stats.paired_t_interval(per_seed)
    lo, hi = stats.conservative_interval((both.lo, both.hi), (t_lo, t_hi))
    p_t = stats.paired_t_p_value(per_seed)
    return {
        "name": name,
        "ours": own,
        "minus": other,
        "seeds": seeds,
        "failed_seeds": sorted(failed_by_arm[own] | failed_by_arm[other]),
        "delta": both.point,
        "ours_sens": float(np.mean(ours_sens)),
        "theirs_sens": float(np.mean(theirs_sens)),
        "lo": lo,
        "hi": hi,
        "lo_bootstrap": both.lo,
        "hi_bootstrap": both.hi,
        "lo_t": t_lo,
        "hi_t": t_hi,
        "p": max(both.p_value, p_t),
        "p_bootstrap": both.p_value,
        "p_t": p_t,
        "width_seeds": by_seeds.hi - by_seeds.lo,
        "width_patients": by_patients.hi - by_patients.lo,
        "per_seed": per_seed,
        "seeds_with_gain": both.seeds_with_gain,
        "paired_sd": statistics.stdev(per_seed),
        "at_val_threshold": _at_val_threshold(
            y, patients, a - cuts(theirs), b - cuts(ours), n_boot
        ),
    }


def analyse(study: Sequence[Entry], codes: dict[str, str], n_boot: int = 2000) -> dict[str, Any]:
    """Run the analysis, for both selection rules.

    Parameters
    ----------
    study : sequence of dict
        From ``load_study``.
    codes : dict
        The code of each arm.
    n_boot : int, default 2000
        Bootstrap replicates per interval.

    Returns
    -------
    dict
        ``failed`` runs, ``primary_rule``, ``cases_behind_threshold``, and under ``rules`` for each
        rule: every seed of every arm, the noise floor from CE, and contrasts P1 to P3.
    """
    arm_of = {code: arm for arm, code in codes.items()}
    by_arm: dict[str, dict[int, Entry]] = {arm: {} for arm in codes}
    failed_by_arm: dict[str, set[int]] = {arm: set() for arm in codes}
    failed = []
    for row in study:
        code, seed = row["config"]["code"], int(row["config"]["seed"])
        if code not in arm_of:
            raise ValueError(f"Unknown arm code {code!r}.")
        if row["status"] == "ok":
            if seed in by_arm[arm_of[code]]:
                raise ValueError(
                    f"Run {row['key']} appears more than once; state which one counts."
                )
            by_arm[arm_of[code]][seed] = row
        else:
            failed.append({"arm": arm_of[code], "seed": seed})
            failed_by_arm[arm_of[code]].add(seed)
    finished = [row for runs in by_arm.values() for row in runs.values()]
    if not finished:
        raise ValueError("No run finished.")
    disease = finished[0]["arrays"]["y_test"]
    patients = finished[0]["arrays"]["patients_test"]
    for row in finished:
        same_labels = np.array_equal(row["arrays"]["y_test"], disease)
        if not same_labels or not np.array_equal(row["arrays"]["patients_test"], patients):
            raise ValueError(f"Run {row['key']} used a different test set.")
    y = (1 - disease).astype(np.float64)
    rules: dict[str, Any] = {}
    for rule in RULES:
        arms = {
            arm: [read_run(runs[seed], rule) for seed in sorted(runs)]
            for arm, runs in by_arm.items()
        }

        def available(own: str, other: str) -> bool:
            return bool(by_arm.get(own)) and bool(by_arm.get(other))

        contrast_of = functools.partial(
            _contrast,
            by_arm=by_arm,
            failed_by_arm=failed_by_arm,
            rule=rule,
            y=y,
            patients=patients,
            n_boot=n_boot,
        )

        contrasts = [
            contrast_of(name, other, own=own)
            for name, own, other in FAMILY
            if available(own, other)
        ]
        unavailable = [
            {"name": name, "ours": own, "minus": other}
            for name, own, other in FAMILY
            if own in by_arm and other in by_arm and not available(own, other)
        ]
        claims = family_claims([(c["lo"], c["hi"], c["p"]) for c in contrasts]) if contrasts else []
        for contrast, claim in zip(contrasts, claims, strict=True):
            primary = rule == PRIMARY_RULE
            contrast["claim"] = claim if primary else NOT_A_CLAIM + claim
        secondary = [
            contrast_of(name, other, own=own)
            for name, own, other in SECONDARY
            if available(own, other)
        ]
        for contrast in secondary:
            label = reanalysis.claim_label(contrast["lo"], contrast["hi"])
            contrast["claim"] = label if rule == PRIMARY_RULE else NOT_A_CLAIM + label
        checks = [
            contrast_of(name, other, own=own)
            for name, own, other in CHECKS
            if available(own, other)
        ]
        for contrast in checks:
            contrast["claim"] = "check, not a claim: " + reanalysis.claim_label(
                contrast["lo"], contrast["hi"]
            )
        ce = [row["sens_roc"] for row in arms.get("ce", [])]
        rules[rule] = {
            "arms": arms,
            "contrasts": contrasts,
            "secondary": secondary,
            "checks": checks,
            "unavailable": unavailable,
            "noise_floor": reanalysis.noise_floor(ce) if len(ce) > 1 else None,
        }
    return {
        "scope": "effusion, NIH ChestX-ray14, end-to-end fine-tuning, 98% specificity on disease",
        "primary_rule": PRIMARY_RULE,
        "failed": sorted(failed, key=lambda f: (f["arm"], f["seed"])),
        "cases_behind_threshold": float(disease.sum()) * (1 - TARGET),
        "rules": rules,
    }


def boost_summary(root: Path, codes: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Return, per arm, the share of positives boosted and the recipients' overlap, per epoch.

    It reads the sealed files, so it runs only after unblinding. Without the share of boosted
    positives and the overlap, a tie could not be read: a control that overlaps the flagged set has
    low contrast by construction.

    Parameters
    ----------
    root : Path
        The results folder.
    codes : dict, optional
        The code of each arm. Two arms can share a loss and differ in setting, so with codes the
        runs are grouped by code; without them, by the loss named in the sealed file.
    """
    arm_of = {code: arm for arm, code in (codes or {}).items()}
    runs: dict[str, list[list[dict[str, Any]]]] = {}
    for path in sorted(root.rglob("sealed/*.json")):
        content = json.loads(path.read_text())
        if "history" in content:
            name = arm_of.get(path.stem.split("_")[1], content["arm"])
            runs.setdefault(name, []).append(content["history"])

    def by_epoch(histories: list[list[dict[str, Any]]], field: str) -> list[float] | None:
        epochs = min(len(history) for history in histories)
        rows = [[history[e][field] for history in histories] for e in range(epochs)]
        if all(value is None for row in rows for value in row):
            return None
        return [float(np.mean([v for v in row if v is not None])) for row in rows]

    return {
        arm: {
            "boosted_by_epoch": by_epoch(histories, "boosted_fraction"),
            "overlap_by_epoch": by_epoch(histories, "recipient_overlap"),
        }
        for arm, histories in runs.items()
    }


def simulate_coverage(
    seeds: int, sims: int, n_boot: int, patients: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Measure how often the intervals hold a known difference, on simulated paired arms.

    Each patient has three images that share a label and a patient effect, and each seed shifts
    the second arm's positives by a different amount, as a training seed would. Sensitivity then has
    a known population value, so the true mean difference is known.

    Returns
    -------
    dict
        ``true_effect`` in points, and the share of simulations in which each interval holds it:
        ``bootstrap``, ``t``, and ``conservative``, the wider limit on each side.
    """
    normal = NormalDist()
    rng = np.random.default_rng(seed)
    cut = normal.inv_cdf(TARGET)
    separation, shift, spread, shared = 1.5, 0.05, 0.05, 0.5
    # The arms' positives have unit variance around their means, so sensitivity is a normal tail.
    draws = rng.normal(shift, spread, 20_000)
    gain = np.array([normal.cdf(float(d) + separation - cut) for d in draws])
    true_effect = 100.0 * (float(gain.mean()) - normal.cdf(separation - cut))
    hits = {"bootstrap": 0, "t": 0, "conservative": 0}
    for _ in range(sims):
        ids = np.repeat(np.arange(patients), 3)
        y = (rng.random(patients) < 0.5).astype(np.float64)[ids]
        person = rng.normal(0, 1.0, patients)[ids]
        a, b = [], []
        for _ in range(seeds):
            noise = rng.normal(0, 1.0, len(ids))
            base = np.sqrt(shared) * person + np.sqrt(1 - shared) * noise + separation * y
            a.append(base)
            b.append(base + rng.normal(shift, spread) * y)
        one, other = np.stack(a), np.stack(b)
        boot = stats.patient_bootstrap(y, ids, one, other, _oriented_sens, n_boot=n_boot)
        per_seed = [_oriented_sens(other[i], y) - _oriented_sens(one[i], y) for i in range(seeds)]
        t = stats.paired_t_interval(per_seed)
        wide = stats.conservative_interval((boot.lo, boot.hi), t)
        hits["bootstrap"] += boot.lo <= true_effect <= boot.hi
        hits["t"] += t[0] <= true_effect <= t[1]
        hits["conservative"] += wide[0] <= true_effect <= wide[1]
    return {"seeds": seeds, "sims": sims, "true_effect": true_effect} | {
        name: count / sims for name, count in hits.items()
    }


def markdown(report: dict[str, Any]) -> str:
    """Return the report's tables as Markdown, in the permitted wording."""
    lines = [
        f"Scope: {report['scope']}.",
        "Sensitivity on no-effusion images at 98% specificity on effusion images, in points.",
        f"Disease cases behind the threshold: {report['cases_behind_threshold']:.0f}.",
        f"Failed runs: {report['failed'] or 'none'}.",
        "",
    ]
    for rule, content in report["rules"].items():
        marker = " (primary)" if rule == report["primary_rule"] else ""
        lines += [f"## Selection rule: {rule}{marker}", ""]
        lines += [
            "| Contrast | Ours minus | Absolute | Difference | 95% interval | Bootstrap "
            "| Paired t | Width: seeds | Width: patients | Seeds with a gain | Failed seeds "
            "| At validation"
            " threshold | Realised specificity difference | Wording |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [_contrast_row(c) for c in content["contrasts"]]
        if content["secondary"]:
            lines += [
                "",
                "Secondary: AUCReshaping at about 6x minus each arm, without Holm correction.",
                "",
                *[_contrast_row(c) for c in content["secondary"]],
            ]
        lines += ["", "Checks: a control below CE would make a gain look like targeting.", ""]
        lines += [_contrast_row(c) for c in content["checks"]]
        if content["unavailable"]:
            lines += ["", f"No finished runs for: {content['unavailable']}."]
        lines += [
            "",
            "| Arm | Seed | Epoch | Test ROC | No Finding only | At validation threshold "
            "| Realised specificity | AUC | pAUC |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for arm, rows in content["arms"].items():
            lines += [
                f"| {arm} | {r['seed']} | {r['epoch']} | {r['sens_roc']:.2f} "
                f"| {_optional(r['sens_no_finding'])} | {r['sens_at_val_threshold']:.2f} "
                f"| {r['realised_specificity']:.2f} "
                f"| {r['auc']:.4f} | {r['pauc']:.4f} |"
                for r in rows
            ]
        lines += ["", f"Noise floor, CE repeats: {content['noise_floor']}", ""]
    if "boost" in report:
        lines += ["## Boosted share and overlap, per epoch", ""]
        for arm, content in report["boost"].items():
            lines.append(
                f"- {arm}: boosted {_fractions(content['boosted_by_epoch'])}; "
                f"overlap {_fractions(content['overlap_by_epoch'])}"
            )
    return "\n".join(lines)


def _contrast_row(c: dict[str, Any]) -> str:
    at = c["at_val_threshold"]
    return (
        f"| {c['name']} | {c['ours']} minus {c['minus']} "
        f"| {c['ours_sens']:.2f} vs {c['theirs_sens']:.2f} | {c['delta']:+.2f} "
        f"| [{c['lo']:+.2f}, {c['hi']:+.2f}] "
        f"| [{c['lo_bootstrap']:+.2f}, {c['hi_bootstrap']:+.2f}] "
        f"| [{c['lo_t']:+.2f}, {c['hi_t']:+.2f}] | {c['width_seeds']:.2f} "
        f"| {c['width_patients']:.2f} | {c['seeds_with_gain']}/{len(c['seeds'])} "
        f"| {c['failed_seeds'] or 'none'} "
        f"| {at['delta']:+.2f} [{at['lo']:+.2f}, {at['hi']:+.2f}] "
        f"| {at['realised_specificity_delta']:+.2f} | {c['claim']} |"
    )


def _optional(value: float | None) -> str:
    return "not saved" if value is None else f"{value:.2f}"


def _fractions(values: list[float] | None) -> str:
    return "not logged" if values is None else ", ".join(f"{v:.2f}" for v in values)


def main() -> None:
    """Check coded runs, run the analysis once, or measure interval coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("integrity")
    check.add_argument("--results", type=Path, required=True)
    check.add_argument("--expected", type=Path)
    run = commands.add_parser("analyse")
    run.add_argument("--results", type=Path, required=True)
    run.add_argument("--codes", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--n-boot", type=int, default=2000)
    coverage = commands.add_parser("coverage")
    coverage.add_argument("--out", type=Path, required=True)
    coverage.add_argument("--sims", type=int, default=200)
    coverage.add_argument("--n-boot", type=int, default=300)
    args = parser.parse_args()
    if args.command == "integrity":
        expected = json.loads(args.expected.read_text()) if args.expected else None
        print(json.dumps(integrity(load_study(args.results), expected), indent=2))
        return
    if args.command == "coverage":
        rows = [simulate_coverage(s, args.sims, args.n_boot, seed=s) for s in (5, 8, 10, 15)]
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        return
    codes = json.loads(args.codes.read_text())
    report = analyse(load_study(args.results), codes, args.n_boot)
    report["boost"] = boost_summary(args.results, codes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.out.write_text(markdown(report))


if __name__ == "__main__":
    main()
