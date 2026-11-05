"""Tables and figures of the synthetic laboratory (one figure per scenario).

    uv run --group bench python -m benchmarks.review.report_e1 S-1 S-2 [...]

    uv run --group bench python -m benchmarks.review.report_e1 --g3
    uv run --group bench python -m benchmarks.review.report_e1 --overview

Writes ``benchmarks/results/figures/s12b/<scenario>*.svg|pdf`` and, per scenario in
``benchmarks/results/s12/e1/<scenario>/``, ``summary.csv`` (final settings, after any grid
extension), ``summary_original.csv`` (the settings before the extension, where one ran) and
``interventions.csv`` (I1-I5). Each line: mean test sensitivity at 0.98 at the selected
checkpoint and delta against CE in points with its two-level 95% CI. ``--g3`` writes
``g3.json`` (gate G3 under the amended criterion, with the original one kept for the
record); ``--overview`` draws every scenario's centre level from the CSVs.
"""

from __future__ import annotations

import csv
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.review import analysis, e1, figures, synthetic
from benchmarks.review.figures import LABELS, Theme

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "benchmarks" / "results" / "figures" / "s12b"
REPORT = ROOT / "dev" / "reports" / "S12b_synthetic.md"
SHOWN = ("ce", "aucreshape", "weighted_bce", "focal")
SCALE = 1.0  # data scale of the runs (tests use smoke-scale runs)
FINAL, ORIGINAL = "chosen.json", "chosen_original.json"
INTERVENTION_LABELS = {
    "aucreshape_pw": "I1: on top of class weights",
    "near_miss": "I2: near misses only",
    "warm_aucreshape": "I3: CE first, then AUCReshaping",
    "hard_negatives": "I4: boost hard negatives",
    "smoothed": "I5: smoothed per-sample boost",
}


@cache
def rows(name: str, chosen: str = FINAL) -> tuple[dict[str, Any], ...]:
    return tuple(e1.current_rows(e1.OUT / name, chosen))  # the chosen setting per arm


@cache
def intervention_rows(name: str) -> tuple[dict[str, Any], ...]:
    return tuple(e1.intervention_rows(e1.OUT / name))


@cache
def labels(name: str, level: Any, seed: int) -> np.ndarray:
    return e1.EXPERIMENTS[name].task(level, seed, SCALE).y_test.numpy()


def _at_level(found: tuple[dict[str, Any], ...], level: Any) -> list[dict[str, Any]]:
    tag = e1.level_tag(level)
    return [r for r in found if r["spec"]["task"].endswith("/" + tag)]


def level_rows(name: str, level: Any, chosen: str = FINAL) -> list[dict[str, Any]]:
    return _at_level(rows(name, chosen), level)


def absolute(
    name: str, level: Any, arm: str, epoch: int | None = None, chosen: str = FINAL
) -> np.ndarray:
    """Per-seed test sensitivity at 0.98 (selected checkpoint unless ``epoch``)."""
    out = []
    found = level_rows(name, level, chosen) + _at_level(intervention_rows(name), level)
    for r in found:
        if r["spec"]["arm"] == arm and r["status"] == "ok":
            seed = r["spec"]["seed"]
            scores = analysis.scores_at(r, epoch or analysis.selected_epoch(r), None)
            out.append(analysis.sens_at_spec(scores, labels(name, level, seed), e1.TARGET))
    return np.array(out)


FIELDS = ["level", "arm", "sens", "sd", "delta", "lo", "hi", "gains"]


def _effects(name: str, arms: tuple[str, ...], chosen: str = FINAL) -> list[dict[str, Any]]:
    """Per level and arm: mean sensitivity, its SD over seeds, and delta against CE."""
    table = []
    for level in e1.EXPERIMENTS[name].levels:
        lr = level_rows(name, level, chosen) + _at_level(intervention_rows(name), level)
        for arm in arms:
            values = absolute(name, level, arm, chosen=chosen)
            row: dict[str, Any] = {
                "level": level,
                "arm": arm,
                "sens": values.mean(),
                "sd": values.std(ddof=1),
            }
            if arm != "ce":
                est = analysis.paired_effect(
                    lr, None, lambda s, lv=level: labels(name, lv, s), "ce", arm, e1.TARGET
                )
                row.update(delta=100 * est.point, lo=100 * est.lo, hi=100 * est.hi,
                           gains=f"{est.seeds_with_gain}/{est.cells}")  # fmt: skip
            table.append(row)
    return table


def _write(table: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)


def summary(name: str, chosen: str = FINAL) -> list[dict[str, Any]]:
    """``summary.csv`` (final settings) or ``summary_original.csv`` (before the extension)."""
    table = _effects(name, e1.EVAL_ARMS, chosen)
    _write(table, e1.OUT / name / ("summary.csv" if chosen == FINAL else "summary_original.csv"))
    return table


def intervention_summary(name: str) -> list[dict[str, Any]]:
    """``interventions.csv``: I1-I5 against the same CE runs."""
    table = _effects(name, ("ce", *e1.INTERVENTIONS))
    _write(table, e1.OUT / name / "interventions.csv")
    return table


def g3() -> dict[str, Any]:
    """Gate G3 on S-1 under the amended criterion, plus the original criterion for the record.

    Amended, post hoc: no arm's best checkpoint mean exceeds the exact
    optimum beyond its 95% interval, and the AUCReshaping-vs-CE interval at the selected and
    the last checkpoint includes 0 or lies below +0.5 points. Original: no checkpoint's
    AUCReshaping-vs-CE interval lies above +0.5 points.
    """
    level = e1.EXPERIMENTS["S-1"].centre
    lr = level_rows("S-1", level)
    optimum = synthetic.linear_tpr(np.eye(e1.D)[0], synthetic.Biomarkers(d=e1.D), 0.02)
    checkpoints = [int(e) for e in lr[0]["spec"]["checkpoints"]]
    best = {}
    for arm in e1.EVAL_ARMS:
        means = []
        for epoch in checkpoints:
            v = absolute("S-1", level, arm, epoch)
            half = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
            means.append((float(v.mean()), float(v.mean() - half), epoch))
        top = max(means)
        best[arm] = {"mean": top[0], "lo": top[1], "epoch": top[2],
                     "exceeds": top[1] > optimum}  # fmt: skip

    def effect(epoch: int | None) -> dict[str, Any]:
        est = analysis.paired_effect(lr, None, lambda s: labels("S-1", level, s), "ce",
                                     "aucreshape", e1.TARGET, epoch=epoch)  # fmt: skip
        return {"delta": 100 * est.point, "lo": 100 * est.lo, "hi": 100 * est.hi}

    selected, last = effect(None), effect(checkpoints[-1])
    per_epoch = {e: effect(e) for e in checkpoints}
    ok = lambda d: d["lo"] <= 0 <= d["hi"] or d["hi"] < 0.5  # noqa: E731
    out = {
        "optimum": optimum,
        "best_checkpoint": best,
        "selected": selected,
        "last": last,
        "per_epoch": per_epoch,
        "amended_pass": not any(b["exceeds"] for b in best.values()) and ok(selected) and ok(last),
        "original_pass": all(d["lo"] <= 0.5 for d in per_epoch.values()),
    }
    (e1.OUT / "S-1" / "g3.json").write_text(json.dumps(out, indent=2))
    return out


def s1(theme: Theme) -> plt.Figure:
    """S-1: when the model is already right, AUCReshaping only gets there sooner."""
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    epochs = np.array(e1.CHECKPOINTS)
    ends = {}
    for arm in ("ce", "aucreshape"):
        values = np.array([absolute("S-1", 1.0, arm, int(e)) for e in epochs])  # (E, seeds)
        colour = theme.roles[arm]
        ax.fill_between(epochs, values.min(1), values.max(1), color=colour, alpha=0.12, lw=0)
        ax.plot(epochs, values.mean(1), color=colour, marker="o", ms=3)
        ends[LABELS[arm]] = float(values.mean(1)[-1])
    optimum = synthetic.linear_tpr(np.eye(e1.D)[0], synthetic.Biomarkers(d=e1.D), 0.02)
    ax.axhline(optimum, color=theme.ink, lw=0.8, ls=(0, (1, 2)))
    ax.text(3, optimum, "best possible (exact)", va="bottom", fontsize=7, color=theme.muted)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.17)
    figures.end_labels(ax, 200, ends, theme, min_gap=0.08)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"test sensitivity at $\mathrm{spec}=0.98$")
    ax.set_title("S-1: the model is already right (10 seeds; band = min--max)")
    fig.tight_layout()
    return fig


def s2(theme: Theme) -> plt.Figure:
    """S-2: headroom of a linear risk score, and how much of it each method captures."""
    levels = e1.EXPERIMENTS["S-2"].levels
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    best = [synthetic.best_linear_tpr(synthetic.Biomarkers(d=e1.D, kappa=k), 0.02, restarts=6)[0]
            for k in levels]  # fmt: skip
    ax.plot(levels, best, color=theme.ink, lw=1.0, ls=(0, (1, 2)), marker="")
    ends = {"best linear score": best[-1]}
    for arm in SHOWN:
        values = [absolute("S-2", k, arm) for k in levels]
        means = np.array([v.mean() for v in values])
        sds = np.array([v.std(ddof=1) for v in values])
        ax.errorbar(levels, means, yerr=sds, color=theme.roles[arm], marker="o", ms=3.5,
                    capsize=2, lw=1.5)  # fmt: skip
        ends[LABELS[arm]] = float(means[-1])
    placebo = np.array([absolute("S-2", k, "placebo").mean() for k in levels])
    ax.plot(levels, placebo, color=theme.roles["aucreshape"], ls=(0, (4, 2)), marker="o", ms=4,
            mfc=theme.surface, lw=1.2)  # fmt: skip
    ends["boost the detected positives (control)"] = float(placebo[-1])
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels, [rf"${k:g}$" for k in levels])
    ax.set_xlim(0.9, 9.5)
    figures.end_labels(ax, 8.4, ends, theme, min_gap=0.07)
    ax.set_xlabel(r"spread ratio $\kappa$ of the diseased group (1 = same spread)")
    ax.set_ylabel(r"test sensitivity at $\mathrm{spec}=0.98$")
    ax.set_title("S-2: disease makes measurements more variable (linear score, 10 seeds)")
    fig.tight_layout()
    return fig


CONTROL_STYLE = {"ls": (0, (4, 2)), "marker": "o", "ms": 4, "lw": 1.2}


def _table(name: str) -> dict[tuple[str, str], dict[str, float]]:
    table = {}
    with (e1.OUT / name / "summary.csv").open() as f:
        for row in csv.DictReader(f):
            values = {
                k: float(v) for k, v in row.items() if k not in {"level", "arm", "gains"} and v
            }
            table[(row["level"], row["arm"])] = values
    return table


def _delta_plot(
    theme: Theme,
    name: str,
    x: list[float],
    xlabel: str,
    title: str,
    xticks: list[str] | None = None,
) -> plt.Figure:
    """delta against CE (points, two-level 95% CI) at each level, one series per arm."""
    table = _table(name)
    levels = [str(level) for level in e1.EXPERIMENTS[name].levels]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.axhline(0, color=theme.muted, lw=0.8)
    ends = {}
    for arm in (*SHOWN[1:], "random_boost"):
        rows = [table[(level, arm)] for level in levels]
        d = np.array([r["delta"] for r in rows])
        err = np.array([[r["delta"] - r["lo"] for r in rows], [r["hi"] - r["delta"] for r in rows]])
        if arm == "random_boost":
            colour, style = theme.roles["aucreshape"], {**CONTROL_STYLE, "mfc": theme.surface}
            label = "same boosts, random positives (control)"
        else:
            colour, style, label = (
                theme.roles[arm],
                {"marker": "o", "ms": 3.5, "lw": 1.5},
                LABELS[arm],
            )
        ax.errorbar(x, d, yerr=err, color=colour, capsize=2, **style)
        ends[label] = float(d[-1])
    ax.set_xticks(x, xticks or [f"{v:g}" for v in x])
    figures.end_labels(ax, x[-1] + 0.03 * (x[-1] - x[0]), ends, theme, min_gap=0.07)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta$ sensitivity vs CE (points)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def s3(theme: Theme) -> plt.Figure:
    x = [100 * v for v in e1.EXPERIMENTS["S-3"].levels]
    return _delta_plot(theme, "S-3", x, r"look-alike negatives (\% of negatives)",
                       "S-3: look-alike negatives decide the threshold (10 seeds)")  # fmt: skip


def s5(theme: Theme) -> plt.Figure:
    x = [100 * v for v in e1.EXPERIMENTS["S-5"].levels]
    return _delta_plot(theme, "S-5", x, r"mislabelled share of the positive labels (\%)",
                       "S-5: mislabelled positives (10 seeds)")  # fmt: skip


def s7(theme: Theme) -> plt.Figure:
    levels = e1.EXPERIMENTS["S-7"].levels
    ticks = [rf"{n} @ {100 * p:g}\%" for n, p in levels]
    return _delta_plot(theme, "S-7", list(range(len(levels))), "training positives @ prevalence",
                       "S-7: rare events (10 seeds)", ticks)  # fmt: skip


def _groups(name: str, level: Any, seed: int) -> np.ndarray:
    return np.asarray(e1.EXPERIMENTS[name].task(level, seed).meta["test_groups"])


def _per_group(row: dict[str, Any], y: np.ndarray, groups: np.ndarray) -> dict[int, float]:
    """Sensitivity of each positive subtype at the threshold that all negatives set."""
    scores = analysis.scores_at(row, analysis.selected_epoch(row), None)
    cut = np.quantile(scores[y < 0.5], e1.TARGET)
    return {g: float((scores[groups == g] > cut).mean()) for g in (0, 1)}


def s4(theme: Theme) -> plt.Figure:
    """Gain on the rare subtype vs the common one, at the operating point set by all negatives."""
    level = e1.EXPERIMENTS["S-4"].centre
    arms = ("aucreshape", "weighted_bce", "focal", "random_boost")
    gains = {arm: {0: [], 1: []} for arm in arms}
    for seed in e1.SEEDS:
        y, groups = labels("S-4", level, seed), _groups("S-4", level, seed)
        by = {r["spec"]["arm"]: r for r in level_rows("S-4", level) if r["spec"]["seed"] == seed}
        base = _per_group(by["ce"], y, groups)
        for arm in arms:
            value = _per_group(by[arm], y, groups)
            for g in (0, 1):
                gains[arm][g].append(100 * (value[g] - base[g]))
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    x = np.arange(len(arms))
    colours = [theme.roles.get(a, theme.roles["aucreshape"]) for a in arms]
    for g, offset, alpha in ((0, -0.18, 0.45), (1, 0.18, 1.0)):
        means = np.array([np.mean(gains[a][g]) for a in arms])
        half = np.array(
            [1.96 * np.std(gains[a][g], ddof=1) / np.sqrt(len(gains[a][g])) for a in arms]
        )
        ax.bar(x + offset, means, width=0.34, color=colours, alpha=alpha, edgecolor=theme.surface,
               linewidth=2)  # fmt: skip
        ax.errorbar(x + offset, means, yerr=half, fmt="none", ecolor=theme.ink, capsize=2, lw=0.8)
    ax.axhline(0, color=theme.muted, lw=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=theme.muted, alpha=a) for a in (0.45, 1.0)]
    ax.legend(handles, ["common subtype (90\\%)", "rare subtype (10\\%)"], loc="upper right")
    ax.set_xticks(x, [LABELS.get(a, "random boost (control)") for a in arms])
    ax.set_ylabel(r"$\Delta$ sensitivity vs CE (points)")
    ax.set_title("S-4: who gains, the rare or the common subtype? (10 seeds, 95\\% CI)")
    fig.tight_layout()
    return fig


def s6(theme: Theme) -> plt.Figure:
    """delta (AUCReshaping - CE) over model size x epochs, and the fraction of positives boosted."""
    exp = e1.EXPERIMENTS["S-6"]
    epochs = list(e1.LONG)
    grid = np.full((len(exp.levels), len(epochs)), np.nan)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    names = []
    for i, level in enumerate(exp.levels):
        model, hidden, n = level
        size = "linear" if model == "linear" else "MLP " + "x".join(map(str, hidden))
        names.append(f"{size}, n={n // 1000}k")
        for j, epoch in enumerate(epochs):
            grid[i, j] = 100 * (absolute("S-6", level, "aucreshape", epoch).mean()
                                - absolute("S-6", level, "ce", epoch).mean())  # fmt: skip
        histories = [
            r["history"] for r in level_rows("S-6", level) if r["spec"]["arm"] == "aucreshape"
        ]
        frac = np.mean([[h["boosted_fraction"] for h in history] for history in histories], axis=0)
        # colour = model, line style = training-set size (two levels must never share a style)
        step = (
            {"MLP 32": 1, "MLP 256x256": 4}
            if theme.name == "light"
            else {"MLP 32": 3, "MLP 256x256": 5}
        )
        colour = theme.muted if size == "linear" else theme.purple_ramp[step[size]]
        style = (0, (4, 2)) if n < 10_000 else "-"
        right.plot(range(1, len(frac) + 1), 100 * frac, lw=1.0, color=colour, ls=style,
                   label=names[-1])  # fmt: skip
    limit = np.nanmax(np.abs(grid))
    image = left.imshow(grid, cmap=figures.diverging(theme), vmin=-limit, vmax=limit, aspect="auto")
    left.set_xticks(range(len(epochs)), [str(e) for e in epochs], rotation=90)
    left.set_yticks(range(len(names)), names)
    left.set_xlabel("epoch")
    left.grid(False)
    fig.colorbar(image, ax=left, label=r"$\Delta$ (points)", shrink=0.8)
    right.set_xlabel("epoch")
    right.set_ylabel(r"training positives boosted (\%)")
    right.set_ylim(0, 100)
    right.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2,
                 handlelength=1.8, columnspacing=0.8)  # fmt: skip
    left.set_title("S-6: does the gain fade with long training?")
    fig.tight_layout()
    return fig


def s8(theme: Theme) -> plt.Figure:
    """False-positive rate reached at the validation threshold when deployment drifts."""
    exp = e1.EXPERIMENTS["S-8"]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    x = [100 * v for v in exp.levels]
    ends = {}
    for arm in ("ce", "aucreshape", "weighted_bce"):
        fpr = []
        for level in exp.levels:
            values = []
            for r in level_rows("S-8", level):
                if r["spec"]["arm"] != arm:
                    continue
                seed, epoch = r["spec"]["seed"], analysis.selected_epoch(r)
                task = exp.task(level, seed)
                with np.load(Path(r["_dir"]) / "scores" / f"{r['key']}.npz") as f:
                    val, test = (
                        f[f"val_{epoch}"].astype(np.float32),
                        f[f"test_{epoch}"].astype(np.float32),
                    )
                cut = np.quantile(val[task.y_val.numpy() < 0.5], e1.TARGET)
                values.append(100 * float((test[task.y_test.numpy() < 0.5] > cut).mean()))
            fpr.append(np.mean(values))
        ax.plot(x, fpr, color=theme.roles[arm], marker="o", ms=3.5)
        ends[LABELS[arm]] = fpr[-1]
    ax.axhline(100 * (1 - e1.TARGET), color=theme.ink, lw=0.8, ls=(0, (1, 2)))
    ax.text(
        x[1], 100 * (1 - e1.TARGET), " intended 2\\%", va="bottom", fontsize=7, color=theme.muted
    )
    ax.set_xticks(x, [f"{v:g}" for v in x])
    figures.end_labels(ax, x[-1] + 0.3, ends, theme, min_gap=0.07)
    ax.set_xlabel(r"look-alikes after deployment (\% of negatives; 2\% in training)")
    ax.set_ylabel(r"false-positive rate reached (\%)")
    ax.set_title("S-8: deployment drift (10 seeds)")
    fig.tight_layout()
    return fig


def _csv(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    table = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            table[(row["level"], row["arm"])] = {
                k: float(v) for k, v in row.items() if k not in {"level", "arm", "gains"} and v
            }
    return table


OVERVIEW = ("aucreshape", *INTERVENTION_LABELS, "weighted_bce")


def overview(theme: Theme) -> plt.Figure:
    """delta vs CE at each scenario's centre level: AUCReshaping, I1-I5 and weighted BCE."""
    names = list(e1.EXPERIMENTS)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6), sharex=True, sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, arm in zip(axes.flat, OVERVIEW, strict=False):
        colour = theme.roles["weighted_bce"] if arm == "weighted_bce" else theme.roles["aucreshape"]
        for yi, name in zip(y, names, strict=True):
            centre = str(e1.EXPERIMENTS[name].centre)
            source = "interventions.csv" if arm in INTERVENTION_LABELS else "summary.csv"
            r = _csv(e1.OUT / name / source)[(centre, arm)]
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=colour, lw=1.2, solid_capstyle="round")
            ax.plot(r["delta"], yi, "o", ms=4.5, color=colour, mec=theme.surface, mew=1.0)
        ax.axvline(0, color=theme.muted, lw=0.8)
        title = INTERVENTION_LABELS.get(arm) or LABELS[arm]
        ax.set_title(title, fontsize=8, loc="left")
        ax.grid(axis="y", visible=False)
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_yticks(y, names)
    for ax in axes[-1, :3]:
        ax.set_xlabel(r"$\Delta$ vs CE (points)")
    axes[0, -1].set_xlabel(r"$\Delta$ vs CE (points)")
    axes[0, -1].xaxis.set_tick_params(labelbottom=True)
    fig.suptitle("Every scenario at its centre level (10 seeds, 95\\% CI)", x=0.01, ha="left",
                 fontsize=9)  # fmt: skip
    fig.tight_layout()
    return fig


DRAW = {"S-1": s1, "S-2": s2, "S-3": s3, "S-4": s4, "S-5": s5, "S-6": s6, "S-7": s7, "S-8": s8}


def _line(name: str, row: dict[str, Any]) -> str:
    text = f"{name} {row['level']!s:>6} {row['arm']:13s} {row['sens']:.4f} +/- {row['sd']:.4f}"
    if row["arm"] != "ce":
        text += f"  delta {row['delta']:+6.2f} [{row['lo']:+6.2f}, {row['hi']:+6.2f}]"
        text += f" ({row['gains']})"
    return text


ROW_NAMES = {
    "aucreshape": "AUCReshaping (final setting)",
    "aucreshape@orig": "AUCReshaping (original setting)",
    "weighted_bce": "weighted BCE (final setting)",
    "weighted_bce@orig": "weighted BCE (original setting)",
    "focal": "focal loss",
    "random_boost": "control: same boosts, random positives",
    "boost_all": "control: boost every positive",
    "placebo": "control: boost detected positives",
    "heldout": "thresholds from validation",
    "oracle": "thresholds from the true distribution",
    **INTERVENTION_LABELS,
}


def _markdown_table(name: str) -> list[str]:
    """Every level and arm of one scenario: delta vs CE [95% CI] (seeds with a gain)."""
    root = e1.OUT / name
    retuned = {"aucreshape", "weighted_bce"}  # the arms a grid extension can change
    sources = [
        ("summary.csv", "", None),
        ("summary_original.csv", "@orig", retuned),
        ("interventions.csv", "", set(INTERVENTION_LABELS)),
    ]
    cells: dict[tuple[str, str], str] = {}
    levels: list[str] = []
    for file, suffix, keep in sources:
        if not (root / file).exists():
            continue
        with (root / file).open() as f:
            for row in csv.DictReader(f):
                if file == "summary.csv" and row["level"] not in levels:
                    levels.append(row["level"])
                if keep is not None and row["arm"] not in keep:
                    continue
                if row["arm"] == "ce":
                    cells[("ce", row["level"])] = f"{100 * float(row['sens']):.1f}%"
                else:
                    cells[(row["arm"] + suffix, row["level"])] = (
                        f"{float(row['delta']):+.2f} [{float(row['lo']):+.2f}, "
                        f"{float(row['hi']):+.2f}] ({row['gains']})"
                    )
    lines = ["| | " + " | ".join(levels) + " |", "|---" * (len(levels) + 1) + "|"]
    for arm in ["ce", *ROW_NAMES]:
        if any((arm, level) in cells for level in levels):
            label = "CE (sensitivity)" if arm == "ce" else ROW_NAMES[arm]
            lines.append(
                f"| {label} | " + " | ".join(cells.get((arm, lv), "") for lv in levels) + " |"
            )
    return lines


def markdown(names: list[str]) -> Path:
    """Write ``REPORT``: every scenario's table and the gate G3 outcome, from the CSVs."""
    out = [
        "# Synthetic laboratory, full tables",
        "",
        "Generated by `python -m benchmarks.review.report_e1 --markdown` from the CSVs in",
        "`benchmarks/results/s12/e1/` (local result data, not versioned). Each cell: delta",
        "sensitivity at 98% specificity against CE in points, [two-level 95% CI over seeds and",
        "test samples], (seeds with a gain, of 10). The summary and the figures are on the",
        "documentation's Results page (`docs/results.md`).",
    ]
    for name in names:
        out += ["", f"## {name}: {e1.EXPERIMENTS[name].situation}", "", *_markdown_table(name)]
    g3_path = e1.OUT / "S-1" / "g3.json"
    if g3_path.exists():
        g = json.loads(g3_path.read_text())
        sel, last = g["selected"], g["last"]
        failing = [e for e, d in g["per_epoch"].items() if d["lo"] > 0.5]
        out += [
            "",
            "## Gate G3 (S-1)",
            "",
            f"Exact optimum: {100 * g['optimum']:.2f}%. AUCReshaping vs CE at the selected "
            f"checkpoint: {sel['delta']:+.2f} [{sel['lo']:+.2f}, {sel['hi']:+.2f}]; at the last "
            f"checkpoint: {last['delta']:+.2f} [{last['lo']:+.2f}, {last['hi']:+.2f}]. "
            f"Amended criterion: {'pass' if g['amended_pass'] else 'fail'}. Original criterion "
            f"(no checkpoint above +0.5 points): {'pass' if g['original_pass'] else 'fail'}"
            + (f" (epochs {', '.join(map(str, failing))})." if failing else "."),
        ]
    REPORT.write_text("\n".join(out) + "\n")
    return REPORT


def main() -> None:
    if sys.argv[1:] == ["--g3"]:
        print(json.dumps(g3(), indent=2))
        return
    if sys.argv[1:] == ["--markdown"]:
        print(markdown(list(e1.EXPERIMENTS)))
        return
    if sys.argv[1:2] == ["--figures"]:  # redraw without recomputing the tables
        for name in sys.argv[2:]:
            for path in figures.render(DRAW[name], name, FIGURES):
                print(path)
        return
    if sys.argv[1:] == ["--overview"]:
        for path in figures.render(overview, "S12b-overview", FIGURES):
            print(path)
        return
    for name in sys.argv[1:]:
        for row in summary(name):
            print(_line(name, row))
        if (e1.OUT / name / ORIGINAL).exists():
            for row in summary(name, ORIGINAL):
                print("original", _line(name, row))
        if (e1.OUT / name / "interventions").exists():
            for row in intervention_summary(name):
                print(_line(name, row))
        if name in DRAW:
            for path in figures.render(DRAW[name], name, FIGURES):
                print(path)


if __name__ == "__main__":
    main()
