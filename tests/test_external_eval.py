from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import src.backend.dataset_import as dataset_import
from src.backend.gallery import Gallery
from src.face_model.fake_model import FakeFaceModel


def write_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def create_registered_root(root: Path, identities: dict[str, tuple[int, int, int]]) -> Path:
    for identity_id, color in identities.items():
        write_image(root / identity_id / "r1.png", np.full((8, 8, 3), fill_value=color, dtype=np.uint8))
    return root


def create_dataset_archive(
    archive_path: Path,
    *,
    test_images: dict[str, tuple[int, int, int]],
    annotations: list[dict],
    registered_images: dict[str, tuple[int, int, int]] | None = None,
) -> Path:
    source_root = archive_path.parent / "source"
    images_root = source_root / "images"
    for image_name, color in test_images.items():
        write_image(images_root / image_name, np.full((8, 8, 3), fill_value=color, dtype=np.uint8))

    if registered_images:
        create_registered_root(source_root / "registered", registered_images)

    annotation_path = source_root / "annotations.jsonl"
    annotation_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in annotations),
        encoding="utf-8",
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as handle:
        for path in source_root.rglob("*"):
            if path.is_file():
                handle.write(path, arcname=str(path.relative_to(source_root)).replace("\\", "/"))
    return archive_path


def test_run_external_eval_uses_local_gallery_when_archive_has_no_registered(tmp_path: Path) -> None:
    local_registered_root = create_registered_root(tmp_path / "local_registered", {"p01": (255, 0, 0)})
    archive_path = create_dataset_archive(
        tmp_path / "local_only.zip",
        test_images={"p01_t01.png": (255, 0, 0)},
        annotations=[
            {
                "image": "p01_t01.png",
                "faces": [{"bbox": [0, 0, 8, 8], "identity": "p01"}],
            }
        ],
    )

    result = dataset_import.run_external_eval(
        archive_path,
        "local",
        model=FakeFaceModel(),
        threshold=0.1,
        local_registered_root=local_registered_root,
    )

    assert result.gallery_source == "local"
    assert result.report.metrics.strict_top1_accuracy == 1.0
    assert result.report.metrics.matched_top1_accuracy == 1.0
    assert result.missed_detections == []
    assert result.false_positives == []


def test_run_external_eval_uses_archive_gallery_when_requested(tmp_path: Path) -> None:
    local_registered_root = create_registered_root(tmp_path / "local_registered", {"p02": (0, 255, 0)})
    archive_path = create_dataset_archive(
        tmp_path / "archive_gallery.zip",
        test_images={"p01_t01.png": (255, 0, 0)},
        annotations=[
            {
                "image": "p01_t01.png",
                "faces": [{"bbox": [0, 0, 8, 8], "identity": "p01"}],
            }
        ],
        registered_images={"p01": (255, 0, 0)},
    )

    result = dataset_import.run_external_eval(
        archive_path,
        "archive",
        model=FakeFaceModel(),
        threshold=0.1,
        local_registered_root=local_registered_root,
    )

    assert result.gallery_source == "archive"
    assert result.report.metrics.strict_top1_accuracy == 1.0
    assert result.report.metrics.confusion_pairs == [("p01", "p01")]


def test_run_external_eval_keeps_local_gallery_when_archive_has_registered_but_local_is_selected(
    tmp_path: Path,
) -> None:
    local_registered_root = create_registered_root(tmp_path / "local_registered", {"p02": (0, 255, 0)})
    archive_path = create_dataset_archive(
        tmp_path / "archive_gallery.zip",
        test_images={"p01_t01.png": (255, 0, 0)},
        annotations=[
            {
                "image": "p01_t01.png",
                "faces": [{"bbox": [0, 0, 8, 8], "identity": "p01"}],
            }
        ],
        registered_images={"p01": (255, 0, 0)},
    )

    result = dataset_import.run_external_eval(
        archive_path,
        "local",
        model=FakeFaceModel(),
        threshold=0.1,
        local_registered_root=local_registered_root,
    )

    assert result.gallery_source == "local"
    assert result.report.metrics.strict_top1_accuracy == 0.0
    assert result.report.metrics.predicted_unknown_precision == 0.0
    assert result.report.metrics.confusion_pairs == [("p01", "unknown")]


def test_run_external_eval_reuses_prebuilt_local_gallery_without_rebuilding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_registered_root = create_registered_root(tmp_path / "local_registered", {"p01": (255, 0, 0)})
    archive_path = create_dataset_archive(
        tmp_path / "local_only.zip",
        test_images={"p01_t01.png": (255, 0, 0)},
        annotations=[
            {
                "image": "p01_t01.png",
                "faces": [{"bbox": [0, 0, 8, 8], "identity": "p01"}],
            }
        ],
    )
    cached_gallery = Gallery()
    cached_gallery.register("p01", [np.array([0.0, 0.0, 1.0], dtype=np.float32)])

    def fail_build_from_dir(self, root, model) -> None:
        raise AssertionError("build_from_dir should not run when local gallery is prebuilt")

    monkeypatch.setattr(Gallery, "build_from_dir", fail_build_from_dir)

    result = dataset_import.run_external_eval(
        archive_path,
        "local",
        model=FakeFaceModel(),
        threshold=0.1,
        local_registered_root=local_registered_root,
        local_gallery=cached_gallery,
    )

    assert result.gallery_source == "local"
    assert result.report.metrics.strict_top1_accuracy == 1.0


def test_run_external_eval_cleans_up_temporary_directory_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = create_dataset_archive(
        tmp_path / "cleanup.zip",
        test_images={"p01_t01.png": (255, 0, 0)},
        annotations=[
            {
                "image": "p01_t01.png",
                "faces": [{"bbox": [0, 0, 8, 8], "identity": "p01"}],
            }
        ],
    )

    captured: dict[str, Path] = {}
    original_extract = dataset_import.extract_dataset_archive

    def wrapped_extract(path: str | Path):
        result = original_extract(path)
        captured["root"] = result.extracted_root
        return result

    monkeypatch.setattr(dataset_import, "extract_dataset_archive", wrapped_extract)

    with pytest.raises(FileNotFoundError):
        dataset_import.run_external_eval(
            archive_path,
            "local",
            model=FakeFaceModel(),
            threshold=0.1,
            local_registered_root=tmp_path / "missing_registered",
        )

    assert "root" in captured
    assert not captured["root"].exists()
