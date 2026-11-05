"""Kaggle kernel template, filled in by benchmarks/kaggle/make_kernels.py.

The kernel checks out the repository at the pinned commit ``SHA``, then runs ``JOBS`` (each a
``python -m`` command line, run from the repository root) on ``WORKERS`` workers, one per GPU
(``CUDA_VISIBLE_DEVICES``), each taking every WORKERS-th job in order. Results go to
``/kaggle/working/results`` and a summary to ``/kaggle/working/kernel_meta.json``. Outputs of
earlier kernels attached as inputs (chaining) are copied into ``results`` first, so finished
runs are skipped and interrupted ones resume. ``AUCRESHAPE_DEADLINE`` makes runs save their
state and stop before Kaggle's 12-hour limit.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SHA = ""  # filled in per kernel
JOBS: list[list[str]] = []  # filled in per kernel
WORKERS = 1  # filled in per kernel
HOURS = 11.0  # leave an hour of margin under Kaggle's 12-hour limit

START = time.time()
REPO = Path("/kaggle/temp/aucreshape")
WORKING = Path("/kaggle/working")
RESULTS = WORKING / "results"


def run(*cmd: str, **kwargs: object) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)  # type: ignore[call-overload]


def gpu_names() -> list[str]:
    """Names of the visible GPUs; empty on CPU machines, which have no nvidia-smi."""
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError:
        return []
    return [name for name in listing.split("\n") if name.strip()]


gpus = gpu_names()
print("GPUs:", gpus, flush=True)

REPO.parent.mkdir(parents=True, exist_ok=True)
run("git", "init", "-q", str(REPO))
run("git", "-C", str(REPO), "fetch", "-q", "--depth", "1",
    "https://github.com/ibouftini/AUCReshaping", SHA)  # fmt: skip
run("git", "-C", str(REPO), "checkout", "-q", "FETCH_HEAD")
run(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO))

for previous in Path("/kaggle/input").glob("*/results"):  # chained outputs
    shutil.copytree(previous, RESULTS, dirs_exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

env = dict(os.environ)
env.update(
    AUCRESHAPE_GIT_SHA=SHA,
    AUCRESHAPE_DATA="/kaggle/temp/data",
    AUCRESHAPE_DEADLINE=str(START + HOURS * 3600),
    PYTHONUNBUFFERED="1",
)
log: list[dict[str, object]] = []


def worker(index: int) -> None:
    for job in JOBS[index::WORKERS]:
        t0 = time.time()
        child_env = dict(env, CUDA_VISIBLE_DEVICES=str(index)) if gpus else env
        done = subprocess.run([sys.executable, "-m", *job], cwd=REPO, env=child_env, check=False)
        log.append({"job": job, "worker": index, "code": done.returncode,
                    "seconds": time.time() - t0})  # fmt: skip
        print(":: done", job, "exit", done.returncode, flush=True)


threads = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
for t in threads:
    t.start()
for t in threads:
    t.join()
meta = {"sha": SHA, "gpus": gpus, "seconds": time.time() - START, "jobs": log}
(WORKING / "kernel_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2), flush=True)
