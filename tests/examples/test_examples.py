"""Run every script in examples/ with --fast. Marker: examples."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.examples
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NEEDS = {"lightning_example.py": "lightning", "hf_trainer_example.py": "transformers"}
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))


def run(cmd: list[str]) -> str:
    done = subprocess.run(cmd, cwd=EXAMPLES, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


@pytest.mark.parametrize("script", [s for s in SCRIPTS if s != "ddp_torchrun.py"])
def test_example_runs(script):
    if script in NEEDS and importlib.util.find_spec(NEEDS[script]) is None:
        pytest.skip(f"{NEEDS[script]} is not installed (uv sync --group examples)")
    assert run([sys.executable, script, "--fast"]).strip()


def test_ddp_example_runs_with_torchrun():
    out = run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "ddp_torchrun.py",
            "--fast",
        ]
    )
    assert "2 ranks agree" in out
