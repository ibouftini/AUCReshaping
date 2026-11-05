"""Write Kaggle kernel folders for a plan (benchmarks/kaggle/plans.py).

    uv run python -m benchmarks.kaggle.make_kernels PLAN OUT_DIR [--sha SHA] [--user USER]

The commit must be pushed to GitHub, since kernels fetch it by SHA (default: HEAD). Then push
each folder with ``kaggle kernels push -p OUT_DIR/<name>``, adding
``--accelerator NvidiaTeslaT4`` for GPU kernels (2 x T4), and gather the outputs with
``python -m benchmarks.kaggle.collect``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

TEMPLATE = Path(__file__).with_name("kernel.py")


@dataclass(frozen=True)
class Kernel:
    name: str  # Kaggle slug: lowercase letters, digits and hyphens
    jobs: list[list[str]]  # ``python -m`` command lines
    gpu: bool = False
    workers: int = 1  # one per GPU on a 2 x T4 machine
    dataset_sources: list[str] = field(default_factory=list)
    kernel_sources: list[str] = field(default_factory=list)  # earlier outputs (chaining)


def write(kernel: Kernel, out: Path, sha: str, user: str) -> Path:
    """Write ``out/<name>/run.py`` and ``kernel-metadata.json``; return the folder."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"sha must be a full 40-character commit hash; got {sha!r}")
    code = TEMPLATE.read_text()
    for marker, value in (
        ('SHA = ""  # filled in per kernel', f'SHA = "{sha}"'),
        ("JOBS: list[list[str]] = []  # filled in per kernel", f"JOBS = {kernel.jobs!r}"),
        ("WORKERS = 1  # filled in per kernel", f"WORKERS = {kernel.workers}"),
    ):
        if marker not in code:
            raise RuntimeError(f"template marker missing: {marker!r}")
        code = code.replace(marker, value)
    folder = out / kernel.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(code)
    meta = {
        "id": f"{user}/{kernel.name}",
        "title": kernel.name,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": kernel.gpu,
        "enable_internet": True,
        "dataset_sources": kernel.dataset_sources,
        "kernel_sources": kernel.kernel_sources,
        "competition_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return folder


def main() -> None:
    from benchmarks.kaggle.plans import PLANS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", choices=sorted(PLANS))
    parser.add_argument("out", type=Path)
    parser.add_argument("--sha")
    parser.add_argument("--user", default=os.environ.get("KAGGLE_USER", "KAGGLE_USER"))
    args = parser.parse_args()
    sha = (
        args.sha
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    for kernel in PLANS[args.plan]:
        folder = write(kernel, args.out, sha, args.user)
        accelerator = " --accelerator NvidiaTeslaT4" if kernel.gpu else ""
        print(f"kaggle kernels push -p {folder}{accelerator}")


if __name__ == "__main__":
    main()
