"""Tests for benchmarks.review.nih (NIH ChestX-ray14 preprocessing) on a tiny fake copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.review import nih

ENTRIES = [
    ("00000001_000.png", "Effusion|Pneumothorax", 1),
    ("00000001_001.png", "No Finding", 1),
    ("00000002_000.png", "Hernia", 2),
    ("00000003_000.png", "Infiltration", 3),
]


def fake_nih(root: Path) -> Path:
    rng = np.random.default_rng(0)
    for i, (name, _, _) in enumerate(ENTRIES):
        folder = root / f"images_00{1 + i % 2}" / "images"
        folder.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 256, size=(40, 40, 4 if i == 0 else 1), dtype=np.uint8)
        mode = "RGBA" if i == 0 else "L"
        Image.fromarray(pixels.squeeze(), mode=mode).save(folder / name)
    lines = ["Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age"]
    lines += [f"{name},{labels},0,{pid},50" for name, labels, pid in ENTRIES]
    (root / "Data_Entry_2017.csv").write_text("\n".join(lines) + "\n")
    (root / "train_val_list.txt").write_text(
        "00000001_000.png\n00000001_001.png\n00000003_000.png\n"
    )
    (root / "test_list.txt").write_text("00000002_000.png\n")
    return root


def test_preprocess_writes_images_and_labels(tmp_path: Path) -> None:
    out = nih.preprocess(fake_nih(tmp_path / "in"), tmp_path / "out", size=16, workers=1)
    images = np.load(out / "images.npy", mmap_mode="r")
    assert images.shape == (4, 16, 16)
    assert images.dtype == np.uint8
    meta = np.load(out / "labels.npz")
    assert list(meta["image"]) == [e[0] for e in ENTRIES]  # sorted by image name
    assert list(meta["patient"]) == [1, 1, 2, 3]
    assert list(meta["split"]) == ["train_val", "train_val", "test", "train_val"]
    columns = list(meta["columns"])
    labels = meta["labels"]
    assert labels.shape == (4, len(nih.FINDINGS) + 1)
    assert labels[0, columns.index("Effusion")] == 1
    assert labels[0, columns.index("Pneumothorax")] == 1
    assert labels[1, columns.index("No Finding")] == 1
    assert labels[1].sum() == 1
    assert labels[2, columns.index("Hernia")] == 1


def test_resizing_matches_pil(tmp_path: Path) -> None:
    root = fake_nih(tmp_path / "in")
    out = nih.preprocess(root, tmp_path / "out", size=16, workers=1)
    images = np.load(out / "images.npy")
    path = root / "images_002" / "images" / "00000001_001.png"
    expected = np.asarray(Image.open(path).convert("L").resize((16, 16), Image.Resampling.BOX))
    np.testing.assert_array_equal(images[1], expected)
