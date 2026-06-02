from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.backend.dataset_import import (
    DatasetArchiveError,
    DatasetLayoutError,
    extract_dataset_archive,
    locate_external_dataset_directory,
)


def _write_archive(archive_path: Path, members: dict[str, bytes]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as handle:
        for name, data in members.items():
            handle.writestr(name, data)


def test_extract_dataset_archive_supports_layout_a_with_optional_registered(tmp_path: Path) -> None:
    archive_path = tmp_path / "layout_a.zip"
    _write_archive(
        archive_path,
        {
            "images/p01_t01.jpg": b"fake-image",
            "annotation.json": b"{}",
            "registered/p01/r1.jpg": b"fake-register",
        },
    )

    result = extract_dataset_archive(archive_path)

    assert result.images_dir.name == "images"
    assert result.annotation_path.name == "annotation.json"
    assert result.annotation_format == "json"
    assert result.registered_dir is not None
    assert result.registered_dir.name == "registered"
    assert result.dataset_root == result.images_dir.parent


def test_extract_dataset_archive_supports_layout_b_nested_under_test_directory(tmp_path: Path) -> None:
    archive_path = tmp_path / "layout_b.zip"
    _write_archive(
        archive_path,
        {
            "test/images/group_01.jpg": b"fake-image",
            "test/annotation.jsonl": b'{"image":"group_01.jpg","faces":[]}\n',
        },
    )

    result = extract_dataset_archive(archive_path)

    assert result.dataset_root.name == "test"
    assert result.images_dir == result.dataset_root / "images"
    assert result.annotation_path == result.dataset_root / "annotation.jsonl"
    assert result.annotation_format == "jsonl"
    assert result.registered_dir is None


def test_extract_dataset_archive_rejects_unsafe_relative_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    _write_archive(
        archive_path,
        {
            "../evil.txt": b"bad",
            "images/p01.jpg": b"fake-image",
            "annotation.json": b"{}",
        },
    )

    with pytest.raises(DatasetArchiveError, match="不安全"):
        extract_dataset_archive(archive_path)


def test_extract_dataset_archive_raises_layout_error_when_structure_is_missing(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad_layout.zip"
    _write_archive(
        archive_path,
        {
            "foo/readme.txt": b"hello",
            "bar/notes.txt": b"world",
        },
    )

    with pytest.raises(DatasetLayoutError, match="未找到 images/ 或标注文件"):
        extract_dataset_archive(archive_path)


def test_extract_dataset_archive_raises_archive_error_for_corrupt_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "broken.zip"
    archive_path.write_bytes(b"not-a-zip")

    with pytest.raises(DatasetArchiveError, match="无法解压"):
        extract_dataset_archive(archive_path)


def test_extract_dataset_archive_falls_back_to_zipfile_when_7z_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "fallback.zip"
    _write_archive(
        archive_path,
        {
            "images/p01_t01.jpg": b"fake-image",
            "annotation.json": b"{}",
        },
    )

    calls: list[str] = []

    def fake_find_7z() -> None:
        return None

    def fake_extract_with_zipfile(zip_path: Path, destination: Path) -> None:
        calls.append(f"{zip_path.name}->{destination.name}")
        with zipfile.ZipFile(zip_path, "r") as handle:
            handle.extractall(destination)

    monkeypatch.setattr("src.backend.dataset_import._find_7z_executable", fake_find_7z)
    monkeypatch.setattr("src.backend.dataset_import._extract_with_zipfile", fake_extract_with_zipfile)

    result = extract_dataset_archive(archive_path)

    assert calls
    assert result.images_dir.exists()


def test_locate_external_dataset_directory_accepts_selected_test_directory(tmp_path: Path) -> None:
    test_root = tmp_path / "test"
    (test_root / "images").mkdir(parents=True)
    (test_root / "annotations.jsonl").write_text('{"image":"group_01.jpg","faces":[]}\n', encoding="utf-8")
    (test_root / "registered" / "p01").mkdir(parents=True)

    result = locate_external_dataset_directory(test_root)

    assert result.dataset_root == test_root
    assert result.images_dir == test_root / "images"
    assert result.annotation_path == test_root / "annotations.jsonl"
    assert result.registered_dir == test_root / "registered"


def test_locate_external_dataset_directory_accepts_dataset_root_with_nested_test_directory(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    test_root = root / "test"
    (test_root / "images").mkdir(parents=True)
    (test_root / "annotation.json").write_text("{}", encoding="utf-8")
    (root / "registered" / "p01").mkdir(parents=True)

    result = locate_external_dataset_directory(root)

    assert result.dataset_root == test_root
    assert result.images_dir == test_root / "images"
    assert result.annotation_path == test_root / "annotation.json"
    assert result.registered_dir == root / "registered"


def test_locate_external_dataset_directory_rejects_parent_directory_without_direct_test_layout(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    test_root = root / "nested" / "test"
    (test_root / "images").mkdir(parents=True)
    (test_root / "annotation.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DatasetLayoutError, match="test/"):
        locate_external_dataset_directory(root)
