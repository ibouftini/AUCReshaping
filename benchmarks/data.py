"""Datasets for the benchmarks. Label 1 is always the class whose sensitivity we want to raise.

Downloads are cached outside the repository, in ``$AUCRESHAPE_DATA`` (default
``~/.cache/aucreshape``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

CACHE = Path(os.environ.get("AUCRESHAPE_DATA", Path.home() / ".cache" / "aucreshape"))


@dataclass
class Split:
    """Train/validation/test tensors. Targets are float, shape (N,) or (N, C)."""

    name: str
    x_train: Tensor
    y_train: Tensor
    x_val: Tensor
    y_val: Tensor
    x_test: Tensor
    y_test: Tensor
    note: str = ""

    @property
    def num_features(self) -> int:
        return int(self.x_train.shape[1])

    def describe(self) -> str:
        def part(y: Tensor) -> str:
            return f"{len(y)} ({int((y > 0.5).sum())} pos)"

        return (
            f"{self.name}: train {part(self.y_train)}, val {part(self.y_val)}, "
            f"test {part(self.y_test)}"
        )


def _stratified_split(
    x: np.ndarray, y: np.ndarray, fractions: tuple[float, float, float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[], [], []]
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        a = round(fractions[0] * len(idx))
        b = a + round(fractions[1] * len(idx))
        parts[0] += idx[:a].tolist()
        parts[1] += idx[a:b].tolist()
        parts[2] += idx[b:].tolist()
    return tuple(rng.permutation(np.array(p)) for p in parts)  # type: ignore[return-value]


def _to_split(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    idx: tuple[np.ndarray, np.ndarray, np.ndarray],
    note: str = "",
) -> Split:
    train, val, test = idx
    mean = x[train].mean(0)
    std = x[train].std(0) + 1e-6

    def t(a: np.ndarray) -> Tensor:
        return torch.tensor(a, dtype=torch.float32)

    xs = [(x[i] - mean) / std for i in (train, val, test)]
    return Split(name, t(xs[0]), t(y[train]), t(xs[1]), t(y[val]), t(xs[2]), t(y[test]), note=note)


def synthetic(
    n: int = 75_000,
    dim: int = 10,
    pos_rate: float = 0.05,
    hard_frac: float = 0.2,
    easy_shift: float = 3.0,
    hard_center: tuple[float, ...] = (0.0, 1.6, 1.6),
    hard_scale: float = 0.3,
    seed: int = 0,
) -> Split:
    """Gaussian data whose high-specificity region hinges on a hard positive sub-cluster.

    Negatives are N(0, I). Most positives ("easy") are N(easy_shift e_1, I). A fraction
    ``hard_frac`` ("hard") forms a tight cluster N(c, hard_scale^2 I) inside the negatives'
    tail (c = hard_center, padded with zeros). It is separable with a curved boundary, but
    rare, so a BCE model trained on imbalanced data under-fits it, and these positives make
    up much of what is misclassified at high specificity.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n * pos_rate)
    n_hard = int(n_pos * hard_frac)
    neg = rng.normal(size=(n - n_pos, dim))
    easy = rng.normal(size=(n_pos - n_hard, dim))
    easy[:, 0] += easy_shift
    center = np.zeros(dim)
    center[: len(hard_center)] = hard_center
    hard = center + hard_scale * rng.normal(size=(n_hard, dim))
    x = np.concatenate([neg, easy, hard]).astype(np.float32)
    y = np.concatenate([np.zeros(len(neg)), np.ones(n_pos)]).astype(np.float32)
    idx = _stratified_split(x, y, (0.4, 0.2, 0.4), seed)
    return _to_split("synthetic", x, y, idx, note=f"{dim}-D Gaussians, {hard_frac:.0%} hard")


# The ULB credit-card fraud dataset (Kaggle "mlg-ulb/creditcardfraud", OpenML data id 1597).
# OpenML's ARFF endpoint has no range support and dropped every download near 140 MiB here,
# so we use the mirror from TensorFlow's imbalanced-data tutorial and pin its content.
_CREDITCARD_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
_CREDITCARD_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


def _creditcard_csv() -> Path:
    """Download the CSV once into the cache and verify its SHA-256."""
    import hashlib
    import shutil
    import urllib.request

    path = CACHE / "creditcard" / "creditcard.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(_CREDITCARD_URL) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _CREDITCARD_SHA256:
        raise RuntimeError(f"{path} does not match the pinned SHA-256; delete it and retry.")
    return path


def creditcard(seed: int = 0, positive: str = "normal") -> Split:
    """ULB credit-card fraud data: 284,807 transactions, 492 frauds (OpenML 1597).

    ``positive="normal"`` follows the paper: normal transactions are label 1 and specificity
    is measured on fraud (the critical class). ``positive="fraud"`` is the usual convention.
    Stratified 60/20/20 split; features standardised with training statistics.
    """
    import pandas as pd

    frame = pd.read_csv(_creditcard_csv())
    if len(frame) != 284_807 or int(frame["Class"].sum()) != 492:
        raise RuntimeError("Unexpected credit-card data: expected 284,807 rows and 492 frauds.")
    x = frame.drop(columns="Class").to_numpy(dtype=np.float64)
    fraud = frame["Class"].to_numpy().astype(np.float32)
    y = 1.0 - fraud if positive == "normal" else fraud
    idx = _stratified_split(x, fraud, (0.6, 0.2, 0.2), seed)
    return _to_split(
        "creditcard", x.astype(np.float32), y, idx, note=f"positive = {positive} transactions"
    )


_MEDMNIST_MD5 = {  # MedMNIST v3 (+) 64x64 files on Zenodo record 10519652
    "breastmnist_64": "742edef2a1fd1524b2efff4bd7ba9364",
    "pneumoniamnist_64": "8f4eceb4ccffa70c672198ea285246c6",
}
# Label of the abnormal class in each MedMNIST task (BreastMNIST: 0 = malignant;
# PneumoniaMNIST: 1 = pneumonia).
_ABNORMAL = {"breastmnist": 0, "pneumoniamnist": 1}


def medmnist(name: str, size: int = 64, positive: str = "normal") -> Split:
    """MedMNIST binary task with its official train/val/test split, images in [0, 1].

    ``positive="normal"`` follows the paper's CXR triage convention: normal images are label 1
    (sensitivity = normal images cleared) and specificity is measured on abnormal images.
    ``positive="abnormal"`` is the usual detection convention.
    """
    import hashlib
    import shutil
    import urllib.request

    stem = f"{name}_{size}"
    path = CACHE / "medmnist" / f"{stem}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://zenodo.org/records/10519652/files/{stem}.npz?download=1"
        with urllib.request.urlopen(url) as response, path.with_suffix(".part").open("wb") as f:
            shutil.copyfileobj(response, f)
        path.with_suffix(".part").rename(path)
    if hashlib.md5(path.read_bytes()).hexdigest() != _MEDMNIST_MD5[stem]:
        raise RuntimeError(f"{path} does not match the published MD5; delete it and retry.")
    arrays = np.load(path)
    abnormal = _ABNORMAL[name]

    def part(split: str) -> tuple[Tensor, Tensor]:
        x = torch.from_numpy(arrays[f"{split}_images"]).float().div(255.0).unsqueeze(1)
        is_abnormal = torch.from_numpy(arrays[f"{split}_labels"][:, 0] == abnormal).float()
        y = 1.0 - is_abnormal if positive == "normal" else is_abnormal
        return x, y

    (xt, yt), (xv, yv), (xs, ys) = part("train"), part("val"), part("test")
    return Split(name, xt, yt, xv, yv, xs, ys, note=f"{size}px, positive = {positive}")
