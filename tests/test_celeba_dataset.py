from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.celeba_dataset import CelebADataset, CelebATestSample, load_celeba_dataset


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_load_celeba_dataset_parses_identity_directories_and_images(tmp_path: Path) -> None:
    root = tmp_path / "celeba_100_identities_3reg_3test"
    write_image(root / "register" / "identity_00070" / "107551.jpg", (255, 0, 0))
    write_image(root / "register" / "identity_00070" / "130995.jpg", (200, 0, 0))
    write_image(root / "register" / "identity_00212" / "000001.jpg", (0, 255, 0))
    write_image(root / "test" / "identity_00070" / "151880.jpg", (255, 0, 0))
    write_image(root / "test" / "identity_00212" / "000002.jpg", (0, 255, 0))
    (root / "register" / "identity_00070" / "notes.txt").write_text("ignore", encoding="utf-8")

    dataset = load_celeba_dataset(root)

    assert isinstance(dataset, CelebADataset)
    assert sorted(dataset.registered_images) == ["identity_00070", "identity_00212"]
    assert len(dataset.registered_images["identity_00070"]) == 2
    assert len(dataset.registered_images["identity_00212"]) == 1
    assert [sample.identity_id for sample in dataset.samples] == [
        "identity_00070",
        "identity_00212",
    ]
    assert [sample.source_path for sample in dataset.samples] == [
        root / "test" / "identity_00070" / "151880.jpg",
        root / "test" / "identity_00212" / "000002.jpg",
    ]
    assert int(np.argmax(dataset.samples[0].face_image[0, 0, :3])) == 2
    assert int(np.argmax(dataset.samples[1].face_image[0, 0, :3])) == 1


def test_load_celeba_dataset_warns_on_empty_identity_and_skips_it(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "celeba_100_identities_3reg_3test"
    write_image(root / "register" / "identity_00070" / "107551.jpg", (255, 0, 0))
    write_image(root / "test" / "identity_00070" / "151880.jpg", (255, 0, 0))
    (root / "register" / "identity_00212").mkdir(parents=True)
    (root / "test" / "identity_00212").mkdir(parents=True)

    with caplog.at_level("WARNING"):
        dataset = load_celeba_dataset(root)

    assert sorted(dataset.registered_images) == ["identity_00070"]
    assert [sample.identity_id for sample in dataset.samples] == ["identity_00070"]
    assert "没有有效图片" in caplog.text


def test_load_celeba_dataset_rejects_missing_directories(tmp_path: Path) -> None:
    root = tmp_path / "celeba_100_identities_3reg_3test"
    (root / "register").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        load_celeba_dataset(root)


def test_load_celeba_dataset_rejects_mismatched_identity_sets(tmp_path: Path) -> None:
    root = tmp_path / "celeba_100_identities_3reg_3test"
    write_image(root / "register" / "identity_00070" / "107551.jpg", (255, 0, 0))
    write_image(root / "test" / "identity_00212" / "151880.jpg", (0, 255, 0))

    with pytest.raises(ValueError):
        load_celeba_dataset(root)
