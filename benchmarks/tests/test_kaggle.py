"""Tests for the Kaggle tooling: kernel generation and output collection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.kaggle import collect, make_kernels, plans

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_generated_kernel_has_metadata_and_pinned_code(tmp_path: Path) -> None:
    kernel = make_kernels.Kernel(
        "aucreshape-test",
        jobs=[["benchmarks.review.nih", "/kaggle/input/data", "/kaggle/working/nih"]],
        gpu=True,
        workers=2,
        dataset_sources=["nih-chest-xrays/data"],
        kernel_sources=["someone/previous-kernel"],
    )
    folder = make_kernels.write(kernel, tmp_path, sha=SHA, user="me")
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "me/aucreshape-test"
    assert meta["enable_gpu"] is True
    assert meta["is_private"] is True
    assert meta["dataset_sources"] == ["nih-chest-xrays/data"]
    assert meta["kernel_sources"] == ["someone/previous-kernel"]
    code = (folder / "run.py").read_text()
    compile(code, "run.py", "exec")
    assert f'SHA = "{SHA}"' in code
    assert "WORKERS = 2" in code
    assert "benchmarks.review.nih" in code


def test_a_short_sha_is_rejected(tmp_path: Path) -> None:
    kernel = make_kernels.Kernel("aucreshape-test", jobs=[["x"]])
    with pytest.raises(ValueError, match="40"):
        make_kernels.write(kernel, tmp_path, sha="abc123", user="me")


def test_every_plan_is_valid() -> None:
    names = [k.name for kernels in plans.PLANS.values() for k in kernels]
    assert len(names) == len(set(names))
    for kernels in plans.PLANS.values():
        for k in kernels:
            assert k.jobs
            assert k.name == k.name.lower()
            assert len(k.name) <= 50


def fake_output(
    root: Path, rows: list[dict[str, object]], sha: str = SHA, block: str = "smoke"
) -> Path:
    folder = root / "results" / block
    (folder / "scores").mkdir(parents=True)
    with (folder / "runs.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            np.savez(folder / "scores" / f"{row['key']}.npz", test_1=np.zeros(3, np.float16))
    meta = {"sha": sha, "gpus": ["Tesla T4", "Tesla T4"], "seconds": 3600.0}
    (root / "kernel_meta.json").write_text(json.dumps(meta))
    return root


def row(key: str, status: str = "ok", sha: str = SHA) -> dict[str, object]:
    return {"key": key, "status": status, "git_sha": sha, "history": []}


def test_merge_adds_rows_scores_and_a_manifest_entry(tmp_path: Path) -> None:
    src = fake_output(tmp_path / "src", [row("a"), row("b", status="failed: nan")])
    summary = collect.merge(src, tmp_path / "into", kernel="me/k")
    lines = (tmp_path / "into" / "runs.jsonl").read_text().splitlines()
    assert [json.loads(line)["key"] for line in lines] == ["a", "b"]
    assert (tmp_path / "into" / "scores" / "a.npz").exists()
    assert summary["new"] == 2
    assert summary["failed"] == 1
    manifest = (tmp_path / "into" / "manifest.jsonl").read_text().splitlines()
    assert json.loads(manifest[0])["kernel"] == "me/k"
    assert json.loads(manifest[0])["gpu_hours"] == pytest.approx(1.0)


def test_merge_is_idempotent(tmp_path: Path) -> None:
    src = fake_output(tmp_path / "src", [row("a")])
    collect.merge(src, tmp_path / "into", kernel="me/k")
    summary = collect.merge(src, tmp_path / "into", kernel="me/k")
    assert summary["new"] == 0
    assert len((tmp_path / "into" / "runs.jsonl").read_text().splitlines()) == 1


def test_conflicting_rows_with_one_key_are_refused(tmp_path: Path) -> None:
    collect.merge(fake_output(tmp_path / "one", [row("a")]), tmp_path / "into", kernel="me/k")
    other = fake_output(tmp_path / "two", [row("a", status="failed: nan")])
    with pytest.raises(ValueError, match="differs"):
        collect.merge(other, tmp_path / "into", kernel="me/k2")


def test_rows_from_another_commit_are_counted(tmp_path: Path) -> None:
    src = fake_output(tmp_path / "src", [row("a", sha="f" * 40)])
    assert collect.merge(src, tmp_path / "into", kernel="me/k")["sha_mismatch"] == 1


def test_gpu_detection_works_on_cpu_machines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression: the template called nvidia-smi unconditionally and crashed CPU kernels.
    import ast
    import subprocess

    tree = ast.parse(make_kernels.TEMPLATE.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gpu_names")
    namespace: dict[str, object] = {"subprocess": subprocess}
    exec(compile(ast.Module([func], []), "kernel.py", "exec"), namespace)
    monkeypatch.setenv("PATH", str(tmp_path))  # no nvidia-smi on this PATH
    assert namespace["gpu_names"]() == []  # type: ignore[operator]


def test_collect_downloads_only_results_metadata_and_logs() -> None:
    import re

    keep = ["kernel_meta.json", "aucreshape-s12e-e4-1.log", "results/e4/ce_s0/runs.jsonl",
            "results/e4/ce_s0/scores/nih_128_swav_resnet50_ce_0_200.npz"]  # fmt: skip
    skip = ["nih256/images.npy", "nih256/labels.npz", "__results__.html"]
    assert all(re.search(collect.SMALL, path) for path in keep)
    assert not any(re.search(collect.SMALL, path) for path in skip)
