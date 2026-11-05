"""Resize NIH ChestX-ray14 once, on Kaggle, for the chest X-ray studies.

    python -m benchmarks.review.nih /kaggle/input/data /kaggle/working/nih256

Input: the Kaggle dataset ``nih-chest-xrays/data`` (``Data_Entry_2017.csv``,
``train_val_list.txt``, ``test_list.txt``, ``images_*/images/*.png``; Wang et al. 2017).
Output, in image-name order:

- ``images.npy``: uint8, shape (N, size, size), grayscale, resized with a box filter;
- ``labels.npz``: ``image`` and ``patient`` ids, ``split`` ("train_val" or "test", the
  official patient-wise lists), ``columns`` (the 14 findings and "No Finding") and
  ``labels``, uint8 of shape (N, 15).

Validation is carved out of "train_val" by patient later, in the loaders.
"""

from __future__ import annotations

import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)


def _resize(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((size, size), Image.Resampling.BOX))


def preprocess(root: Path, out: Path, size: int = 256, workers: int = 4) -> Path:
    import pandas as pd

    entries = pd.read_csv(root / "Data_Entry_2017.csv").sort_values("Image Index")
    paths = {p.name: p for p in root.glob("images_*/images/*.png")}
    names = entries["Image Index"].tolist()
    missing = [n for n in names if n not in paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:3]}")
    test = set((root / "test_list.txt").read_text().split())
    columns = [*FINDINGS, "No Finding"]
    labels = np.zeros((len(names), len(columns)), dtype=np.uint8)
    for i, found in enumerate(entries["Finding Labels"]):
        for name in found.split("|"):
            labels[i, columns.index(name)] = 1

    out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        out / "images.npy", mode="w+", dtype=np.uint8, shape=(len(names), size, size)
    )
    resize = partial(_resize, size=size)
    if workers > 1:
        with Pool(workers) as pool:
            for i, array in enumerate(pool.imap(resize, [paths[n] for n in names], 64)):
                images[i] = array
    else:
        for i, name in enumerate(names):
            images[i] = resize(paths[name])
    images.flush()
    np.savez(
        out / "labels.npz",
        image=np.array(names),
        patient=entries["Patient ID"].to_numpy(),
        split=np.array(["test" if n in test else "train_val" for n in names]),
        columns=np.array(columns),
        labels=labels,
    )
    return out


def find_root(root: Path) -> Path:
    """``root`` if it holds the index file, else the first folder below it that does."""
    if (root / "Data_Entry_2017.csv").exists():
        return root
    return next(root.parent.rglob("Data_Entry_2017.csv")).parent


if __name__ == "__main__":
    preprocess(find_root(Path(sys.argv[1])), Path(sys.argv[2]))
