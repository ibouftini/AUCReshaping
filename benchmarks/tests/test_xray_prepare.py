"""Preparing NIH ChestX-ray14 and the pretrained weights for the new chest X-ray study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from benchmarks.review import xray_prepare as prep

ROWS = [
    ("a_000.png", "Effusion", 1),
    ("b_000.png", "No Finding", 2),
    ("c_000.png", "Effusion|Mass", 3),
    ("d_000.png", "Pneumothorax", 4),
    ("e_000.png", "No Finding", 5),
]


def _fake_nih(root: Path, test: tuple[str, ...] = ("d_000.png", "e_000.png")) -> Path:
    folder = root / "images_001" / "images"
    folder.mkdir(parents=True)
    lines = ["Image Index,Finding Labels,Patient ID"]
    for i, (name, found, patient) in enumerate(ROWS):
        Image.fromarray(np.full((16, 16), 10 * (i + 1), dtype=np.uint8)).save(folder / name)
        lines.append(f"{name},{found},{patient}")
    (root / "Data_Entry_2017.csv").write_text("\n".join(lines) + "\n")
    (root / "test_list.txt").write_text("\n".join(test) + "\n")
    return root


def test_test_data_goes_to_separate_files(tmp_path: Path) -> None:
    out = prep.prepare_images(_fake_nih(tmp_path / "nih"), tmp_path / "out", size=8, workers=1)
    train = np.load(out / "images_trainval.npy", mmap_mode="r")
    test = np.load(out / "images_test.npy", mmap_mode="r")
    assert train.shape == (3, 8, 8)
    assert test.shape == (2, 8, 8)
    with np.load(out / "labels_trainval.npz") as saved:
        assert saved["image"].tolist() == ["a_000.png", "b_000.png", "c_000.png"]
        assert saved["patient"].tolist() == [1, 2, 3]
        columns = saved["columns"].tolist()
        assert saved["labels"][:, columns.index("Effusion")].tolist() == [1, 0, 1]
    with np.load(out / "labels_test.npz") as saved:
        assert saved["image"].tolist() == ["d_000.png", "e_000.png"]
    # The images stay matched with their labels.
    assert int(train[0, 0, 0]) == 10
    assert int(test[0, 0, 0]) == 40


def test_a_patient_in_both_splits_is_refused(tmp_path: Path) -> None:
    root = _fake_nih(tmp_path / "nih")
    text = (
        (root / "Data_Entry_2017.csv")
        .read_text()
        .replace("e_000.png,No Finding,5", "e_000.png,No Finding,1")
    )
    (root / "Data_Entry_2017.csv").write_text(text)
    with pytest.raises(ValueError, match="patient"):
        prep.prepare_images(root, tmp_path / "out", size=8, workers=1)


def test_a_missing_image_is_refused(tmp_path: Path) -> None:
    root = _fake_nih(tmp_path / "nih")
    (root / "images_001" / "images" / "a_000.png").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        prep.prepare_images(root, tmp_path / "out", size=8, workers=1)


def test_normalise_matches_the_documented_range() -> None:
    x = torch.tensor([0, 255], dtype=torch.uint8)
    out = prep.normalise(x)
    assert out.tolist() == pytest.approx([-1024.0, 1024.0])
    assert out.dtype == torch.float32


def test_backbone_takes_one_channel_and_gives_the_requested_outputs() -> None:
    model = prep.densenet(num_outputs=1)
    out = model(torch.zeros(2, 1, 64, 64))
    assert out.shape == (2, 1)


def test_plain_state_keeps_only_what_the_network_needs() -> None:
    model = prep.densenet(num_outputs=18)
    state = dict(model.state_dict())
    state["op_threshs"] = torch.zeros(18)
    state["upsample.scale"] = torch.zeros(1)
    plain = prep.plain_state(state)
    assert "op_threshs" not in plain
    assert "upsample.scale" not in plain
    fresh = prep.densenet(num_outputs=18)
    fresh.load_state_dict(plain, strict=True)
    x = torch.rand(1, 1, 64, 64)
    model.eval()
    fresh.eval()
    assert torch.equal(model(x), fresh(x))


def test_backbone_from_state_replaces_the_head_and_keeps_the_features() -> None:
    source = prep.densenet(num_outputs=18)
    plain = prep.plain_state(source.state_dict())
    model = prep.backbone_from_state(plain, seed=1)
    assert model.classifier.out_features == 1
    assert torch.equal(model.features.conv0.weight, source.features.conv0.weight)
    other = prep.backbone_from_state(plain, seed=2)
    assert not torch.equal(model.classifier.weight, other.classifier.weight)
    again = prep.backbone_from_state(plain, seed=1)
    assert torch.equal(model.classifier.weight, again.classifier.weight)


def test_only_eligible_weights_are_prepared() -> None:
    # W1: weights trained on NIH, or on images taken from it, can never be used.
    assert set(prep.WEIGHTS) == {"chex", "mimic_nb", "mimic_ch", "pc"}
    for name, file in prep.WEIGHTS.items():
        assert file.startswith(f"{name}-densenet121")
        assert "nih" not in file
        assert "kaggle" not in file


def test_kernel_is_private_with_internet_for_the_one_download(tmp_path: Path) -> None:
    folder = prep.make_kernel(tmp_path, user="someone")
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/aucreshape-s13c-prepare"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is True
    assert meta["dataset_sources"] == ["nih-chest-xrays/data"]
    compile((folder / meta["code_file"]).read_text(), "kernel", "exec")
