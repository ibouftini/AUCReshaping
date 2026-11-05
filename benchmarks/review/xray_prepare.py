"""Prepare NIH ChestX-ray14 and the pretrained weights for the new chest X-ray study.

    python -m benchmarks.review.xray_prepare make --user USER --out DIR

It runs once, as a private Kaggle script kernel, and later kernels read its output, so they need no
internet and no extra package. The file is self-contained and is pushed verbatim.

Test images and labels go to their own files, so a pilot or tuning kernel never opens them, which is
stronger than promising not to read them.

The weights come from TorchXRayVision (Cohen et al.). Only those trained on a single dataset other
than NIH are eligible: NIH-trained weights may have seen the test images, and the training positives
would already be fitted, so almost nothing would be boosted. ``rsna`` is excluded too, because its
images are a subset of NIH. Each set is checked against the package's own model before it is saved.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import Tensor, nn

KERNEL_RUN: bool = False

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
XRV_VERSION = "1.5.7"
_SUFFIX = "densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
WEIGHTS = {name: f"{name}-{_SUFFIX}" for name in ("chex", "mimic_nb", "mimic_ch", "pc")}


def _resize(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR))


def _write_split(
    out: Path, split: str, names: list[str], paths: dict[str, Path], size: int, workers: int
) -> None:
    images = np.lib.format.open_memmap(
        out / f"images_{split}.npy", mode="w+", dtype=np.uint8, shape=(len(names), size, size)
    )
    resize = partial(_resize, size=size)
    files = [paths[name] for name in names]
    if workers > 1:
        with Pool(workers) as pool:
            for i, array in enumerate(pool.imap(resize, files, 64)):
                images[i] = array
    else:
        for i, path in enumerate(files):
            images[i] = resize(path)
    images.flush()


def prepare_images(root: Path, out: Path, size: int = 224, workers: int = 4) -> Path:
    """Resize every image once, keeping the official test list in separate files.

    Parameters
    ----------
    root : Path
        The folder holding ``Data_Entry_2017.csv``, ``test_list.txt`` and ``images_*/images/*.png``.
    out : Path
        Receives ``images_{trainval,test}.npy``, uint8 of shape (N, size, size), and
        ``labels_{trainval,test}.npz`` with ``image``, ``patient``, ``columns`` and ``labels``.
    size : int, default 224
        The side of the square images.
    workers : int, default 4
        Decoding processes.

    Returns
    -------
    Path
        ``out``.
    """
    import pandas as pd

    entries = pd.read_csv(root / "Data_Entry_2017.csv").sort_values("Image Index")
    paths = {path.name: path for path in root.glob("images_*/images/*.png")}
    names = entries["Image Index"].tolist()
    missing = [name for name in names if name not in paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:3]}")
    listed = set((root / "test_list.txt").read_text().split())
    is_test = np.array([name in listed for name in names])
    patients = entries["Patient ID"].to_numpy()
    shared = set(patients[is_test].tolist()) & set(patients[~is_test].tolist())
    if shared:
        raise ValueError(
            f"{len(shared)} patient(s) appear in both splits, e.g. {sorted(shared)[:3]}"
        )
    columns = [*FINDINGS, "No Finding"]
    labels = np.zeros((len(names), len(columns)), dtype=np.uint8)
    for i, found in enumerate(entries["Finding Labels"]):
        for name in found.split("|"):
            labels[i, columns.index(name)] = 1
    out.mkdir(parents=True, exist_ok=True)
    for split, chosen in (("trainval", ~is_test), ("test", is_test)):
        selected = [name for name, keep in zip(names, chosen, strict=True) if keep]
        _write_split(out, split, selected, paths, size, workers)
        np.savez(
            out / f"labels_{split}.npz",
            image=np.array(selected),
            patient=patients[chosen],
            columns=np.array(columns),
            labels=labels[chosen],
        )
    return out


def normalise(x: Tensor) -> Tensor:
    """Scale uint8 images to [-1024, 1024], the range these weights expect (any shape)."""
    return (2.0 * (x.float() / 255.0) - 1.0) * 1024.0


def densenet(num_outputs: int) -> nn.Module:
    """Return a DenseNet-121 for one-channel images; inputs are (N, 1, H, W), outputs (N, K)."""
    model = torchvision.models.densenet121(num_classes=num_outputs)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def plain_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Keep the network's weights, dropping the extras the package stores with them."""
    return {
        key: value for key, value in state.items() if key.startswith(("features.", "classifier."))
    }


def backbone_from_state(state: dict[str, Tensor], seed: int) -> nn.Module:
    """Load pretrained features under a new one-output head, whose initialization ``seed`` sets."""
    model = densenet(num_outputs=1)
    features = {key: value for key, value in state.items() if key.startswith("features.")}
    missing, unexpected = model.load_state_dict(features, strict=False)
    if unexpected or set(missing) != {"classifier.weight", "classifier.bias"}:
        raise ValueError(f"Unexpected keys: missing {missing}, unexpected {unexpected}.")
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.classifier.weight.copy_(
            torch.randn(model.classifier.weight.shape, generator=generator) * 0.01
        )
        model.classifier.bias.zero_()
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_weights(out: Path) -> dict[str, Any]:
    """Install the package once, verify each set of weights against it, and save plain files."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"torchxrayvision=={XRV_VERSION}"],
        check=True,
    )
    import torchxrayvision as xrv

    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"package": xrv.__version__, "weights": {}}
    x = torch.rand(2, 1, 224, 224) * 2048 - 1024
    for name, file in WEIGHTS.items():
        source = xrv.models.DenseNet(weights=f"densenet121-res224-{name}", apply_sigmoid=False)
        source.eval()
        state = plain_state(source.state_dict())
        model = densenet(num_outputs=state["classifier.weight"].shape[0])
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            difference = float((model(x) - source.classifier(source.features2(x))).abs().max())
        if difference > 1e-3:
            raise ValueError(f"{name}: differs from the package's model by {difference}.")
        downloaded = next(Path.home().rglob(file))
        target = out / f"{name}.pt"
        torch.save(state, target)
        report["weights"][name] = {
            "file": file,
            "sha256": _sha256(downloaded),
            "plain_sha256": _sha256(target),
            "difference": difference,
            "pathologies": list(source.pathologies),
            "effusion_index": list(source.pathologies).index("Effusion"),
            "dropout": any(m.p > 0 for m in model.modules() if isinstance(m, nn.Dropout)),
            "drop_rate": source.drop_rate if hasattr(source, "drop_rate") else 0.0,
        }
    return report


def run_kernel() -> None:
    """Prepare the images and the weights in the kernel's output folder."""
    out = Path("/kaggle/working")
    report: dict[str, Any] = {"internet": True, "cpus": os.cpu_count()}
    try:
        report |= prepare_weights(out / "weights")
    except Exception as error:  # the images are still worth having, and this says what to do next
        report.update(internet=False, weights_error=repr(error))
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report)[:2000], flush=True)
    index = sorted(glob.glob("/kaggle/input/**/Data_Entry_2017.csv", recursive=True))[0]
    prepare_images(Path(index).parent, out / "nih224", size=224, workers=os.cpu_count() or 4)
    for split in ("trainval", "test"):
        with np.load(out / "nih224" / f"labels_{split}.npz") as saved:
            report[f"images_{split}"] = int(saved["labels"].shape[0])
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "weights"}), flush=True)


def make_kernel(root: Path, user: str) -> Path:
    """Write the private CPU kernel, with internet for its one download; return its folder."""
    slug = "aucreshape-s13c-prepare"
    folder = root / slug
    folder.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).read_text()
    code = code.replace("\nKERNEL_RUN: bool = False\n", "\nKERNEL_RUN: bool = True\n", 1)
    (folder / "xray_prepare.py").write_text(code)
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "xray_prepare.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["nih-chest-xrays/data"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


def main() -> None:
    """Run the kernel, or write it."""
    if KERNEL_RUN:
        run_kernel()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make")
    make.add_argument("--user", default=os.environ.get("KAGGLE_USER", "KAGGLE_USER"))
    make.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(make_kernel(args.out, args.user))


if __name__ == "__main__":
    main()
