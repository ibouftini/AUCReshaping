"""Dataset registry of the evaluation.

Each loader returns a :class:`benchmarks.review.runner.Task` with label 1 = the class whose
sensitivity is raised (fraud, failure, pulsar, minority), float32 features imputed with
training medians and standardised with training statistics. Official splits are kept where
they exist; otherwise the split is stratified 60/20/20 with ``split_seed``.

Files are cached under ``benchmarks.data.CACHE`` and checked against a pinned SHA-256.
Nothing is redistributed: every source keeps its own licence (``SOURCES``). Sources on Kaggle
are fetched with the authenticated Kaggle CLI; sources above ~300 MB are processed on
Kaggle, not downloaded here (the home connection is slow).
"""

from __future__ import annotations

import hashlib
import io
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from benchmarks import data
from benchmarks.review.runner import Task


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str
    licence: str
    citation: str
    domain: str


_UCI = "https://archive.ics.uci.edu/static/public/"
SOURCES = {
    "creditcard": Source(
        data._CREDITCARD_URL,
        data._CREDITCARD_SHA256,
        "DbCL v1.0 (Kaggle mlg-ulb/creditcardfraud)",
        "Dal Pozzolo et al. (2015), Calibrating probability with undersampling, IEEE SSCI",
        "finance: card fraud",
    ),
    "aps": Source(
        _UCI + "421/aps+failure+at+scania+trucks.zip",
        "5504d0402f54faaf97ac0ca085a621645763f5cfea2eb29c592b057d43d4db89",
        "GPL-3.0 (file header); UCI lists CC BY 4.0",
        "Scania CV AB (2016), APS Failure at Scania Trucks, UCI ML Repository",
        "industrial: predictive maintenance",
    ),
    "secom": Source(
        _UCI + "179/secom.zip",
        "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e",
        "CC BY 4.0 (UCI)",
        "McCann & Johnston (2008), SECOM, UCI ML Repository",
        "industrial: semiconductor process",
    ),
    "htru2": Source(
        _UCI + "372/htru2.zip",
        "ba442c076dd22a8952700f26e38499fc1806037dcf7bea0e125e6bfba393f379",
        "CC BY 4.0 (UCI)",
        "Lyon et al. (2016), Fifty years of pulsar candidate selection, MNRAS 459",
        "science: astronomy",
    ),
    "imblearn": Source(
        "imblearn.datasets.fetch_datasets (Zenodo record 61452)",
        "",  # checked by imbalanced-learn itself
        "per dataset (UCI, LIBSVM, KEEL origins)",
        "Lemaitre, Nogueira & Aridas (2017), imbalanced-learn, JMLR 18",
        "breadth: 27 benchmark datasets",
    ),
}
IMBLEARN = [
    "ecoli",
    "optical_digits",
    "satimage",
    "pen_digits",
    "abalone",
    "sick_euthyroid",
    "spectrometer",
    "car_eval_34",
    "isolet",
    "us_crime",
    "yeast_ml8",
    "scene",
    "libras_move",
    "thyroid_sick",
    "coil_2000",
    "arrhythmia",
    "solar_flare_m0",
    "oil",
    "car_eval_4",
    "wine_quality",
    "letter_img",
    "yeast_me2",
    "webpage",
    "ozone_level",
    "mammography",
    "protein_homo",
    "abalone_19",
]


def _archive(name: str) -> Path:
    source = SOURCES[name]
    path = data.CACHE / name / f"{name}.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(source.url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.rename(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source.sha256:
        raise RuntimeError(f"{path} does not match the pinned SHA-256; delete it and retry.")
    return path


def available(name: str) -> bool:
    """Whether ``name`` can be loaded without a download."""
    if name == "creditcard":
        return (data.CACHE / "creditcard" / "creditcard.csv").exists()
    if name in IMBLEARN:
        index = IMBLEARN.index(name) + 1
        return (data.CACHE / "imblearn" / "zenodo" / f"x{index}data.npz").exists()
    return (data.CACHE / name / f"{name}.zip").exists()


def _task(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    idx: tuple[np.ndarray, np.ndarray, np.ndarray],
    domain: str,
) -> Task:
    """Median-impute and standardise with training statistics, then build the Task."""
    train, val, test = idx
    x = x.astype(np.float64)
    median = np.nanmedian(x[train], axis=0)
    median = np.where(np.isnan(median), 0.0, median)  # columns missing in all of train
    x = np.where(np.isnan(x), median, x)
    mean, std = x[train].mean(0), x[train].std(0)
    x = (x - mean) / np.where(std > 0, std, 1.0)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32)

    return Task(
        name,
        t(x[train]),
        t(y[train]),
        t(x[val]),
        t(y[val]),
        t(x[test]),
        t(y[test]),
        meta={"domain": domain},
    )


def creditcard(split_seed: int = 0) -> Task:
    import pandas as pd

    frame = pd.read_csv(data._creditcard_csv())
    x = frame.drop(columns="Class").to_numpy(dtype=np.float64)
    y = frame["Class"].to_numpy().astype(np.float32)  # fraud = 1
    idx = data._stratified_split(x, y, (0.6, 0.2, 0.2), split_seed)
    return _task("creditcard", x, y, idx, SOURCES["creditcard"].domain)


def aps(split_seed: int = 0) -> Task:
    """Official test set; the official training set is split 80/20 into train and val."""
    import pandas as pd

    with zipfile.ZipFile(_archive("aps")) as z:
        frames = [
            pd.read_csv(z.open(f"aps_failure_{part}_set.csv"), skiprows=20, na_values="na")
            for part in ("training", "test")
        ]
    xs = [f.drop(columns="class").to_numpy(dtype=np.float64) for f in frames]
    ys = [(f["class"] == "pos").to_numpy().astype(np.float32) for f in frames]
    train, val, _ = data._stratified_split(xs[0], ys[0], (0.8, 0.2, 0.0), split_seed)
    test = np.arange(len(ys[1])) + len(ys[0])
    x, y = np.concatenate(xs), np.concatenate(ys)
    return _task("aps", x, y, (train, val, test), SOURCES["aps"].domain)


def secom(split_seed: int = 0) -> Task:
    with zipfile.ZipFile(_archive("secom")) as z:
        x = np.loadtxt(z.open("secom.data"))
        labels = np.array([int(line.split()[0]) for line in z.open("secom_labels.data")])
    y = (labels == 1).astype(np.float32)  # 1 = fail
    idx = data._stratified_split(x, y, (0.6, 0.2, 0.2), split_seed)
    return _task("secom", x, y, idx, SOURCES["secom"].domain)


def htru2(split_seed: int = 0) -> Task:
    with zipfile.ZipFile(_archive("htru2")) as z:
        text = z.read("HTRU_2.csv").decode().replace("\r", "\n")  # old Mac line endings
    table = np.loadtxt(io.StringIO(text), delimiter=",")
    x, y = table[:, :-1], table[:, -1].astype(np.float32)  # 1 = pulsar
    idx = data._stratified_split(x, y, (0.6, 0.2, 0.2), split_seed)
    return _task("htru2", x, y, idx, SOURCES["htru2"].domain)


def imblearn(name: str, split_seed: int = 0) -> Task:
    from imblearn.datasets import fetch_datasets

    bunch = fetch_datasets(data_home=str(data.CACHE / "imblearn"), filter_data=(name,))[name]
    x = np.asarray(bunch.data, dtype=np.float64)
    y = (np.asarray(bunch.target) == 1).astype(np.float32)
    idx = data._stratified_split(x, y, (0.6, 0.2, 0.2), split_seed)
    return _task(name, x, y, idx, SOURCES["imblearn"].domain)


LOADERS: dict[str, Callable[..., Task]] = {
    "creditcard": creditcard,
    "aps": aps,
    "secom": secom,
    "htru2": htru2,
}


def load(name: str, split_seed: int = 0) -> Task:
    if name in LOADERS:
        return LOADERS[name](split_seed)
    if name in IMBLEARN:
        return imblearn(name, split_seed)
    raise ValueError(f"unknown dataset {name!r}")
