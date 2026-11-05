"""Download finished Kaggle kernels and merge their results into the local results tree.

    uv run python -m benchmarks.kaggle.collect INTO KERNEL [KERNEL ...]

For each kernel (``user/slug``), the output is downloaded with the authenticated Kaggle CLI
(``KAGGLE_CLI``, or ``kaggle`` on the path), then every ``runs.jsonl`` under its ``results``
folder is
merged into ``INTO/runs.jsonl`` (rows deduplicated by key; a key whose row differs is an
error), score files are copied to ``INTO/scores`` (git-ignored), and one line per kernel is
appended to ``INTO/manifest.jsonl`` (commit, GPUs, hours, rows, failures). Rows produced by
another commit than the kernel's are counted, not dropped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KAGGLE = os.environ.get("KAGGLE_CLI") or shutil.which("kaggle") or "kaggle"
# Results, metadata and logs only: never the large arrays a kernel may also leave in its output
# (the s12a preprocessing kernel holds a 7 GB images.npy; home bandwidth is ~0.35 MB/s).
SMALL = r"(^kernel_meta\.json$|\.log$|^results/)"


def merge(src: Path, into: Path, kernel: str) -> dict[str, Any]:
    """Merge one kernel's output folder into ``into``; return the manifest entry."""
    into.mkdir(parents=True, exist_ok=True)
    (into / "scores").mkdir(exist_ok=True)
    runs = into / "runs.jsonl"
    known = {}
    if runs.exists():
        for line in runs.read_text().splitlines():
            known[json.loads(line)["key"]] = line
    meta_path = src / "kernel_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    new, failed, mismatch, seen = [], 0, 0, 0
    for path in sorted(src.glob("results/**/runs.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            seen += 1
            failed += row["status"] != "ok"
            mismatch += bool(meta.get("sha")) and row.get("git_sha") != meta["sha"]
            if row["key"] in known:
                if json.loads(known[row["key"]]) != row:
                    raise ValueError(f"row {row['key']} differs from the one already merged")
                continue
            known[row["key"]] = line
            new.append(line)
            scores = path.parent / "scores" / f"{row['key']}.npz"
            if scores.exists():
                shutil.copy2(scores, into / "scores" / scores.name)
    with runs.open("a") as f:
        f.writelines(line + "\n" for line in new)
    entry = {
        "kernel": kernel,
        "collected": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha": meta.get("sha"),
        "gpus": meta.get("gpus"),
        "gpu_hours": meta.get("seconds", 0.0) / 3600,
        "rows": seen,
        "new": len(new),
        "failed": failed,
        "sha_mismatch": mismatch,
    }
    with (into / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def main() -> None:
    into, kernels = Path(sys.argv[1]), sys.argv[2:]
    for kernel in kernels:
        status = subprocess.run(
            [KAGGLE, "kernels", "status", kernel], capture_output=True, text=True, check=True
        ).stdout
        if "COMPLETE" not in status:
            print(f"{kernel}: not complete ({status.strip()})")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([KAGGLE, "kernels", "output", kernel, "-p", tmp,
                            "--file-pattern", SMALL], check=True)  # fmt: skip
            print(kernel, merge(Path(tmp), into, kernel))


if __name__ == "__main__":
    main()
