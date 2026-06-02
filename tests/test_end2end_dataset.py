from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.end2end_dataset import load_grouped_self_dataset


def write_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def create_roots(root: Path) -> tuple[Path, Path]:
    registered_root = root / "registered"
    test_root = root / "test"
    write_image(
        registered_root / "p01" / "r1.png",
        np.full((8, 8, 3), fill_value=(255, 0, 0), dtype=np.uint8),
    )
    write_image(
        test_root / "images" / "group_01.jpg",
        np.full((8, 8, 3), fill_value=(0, 255, 0), dtype=np.uint8),
    )
    write_image(
        test_root / "images" / "p03_t01.jpg",
        np.full((8, 8, 3), fill_value=(0, 0, 255), dtype=np.uint8),
    )
    return registered_root, test_root


def test_load_grouped_self_dataset_supports_json_and_jsonl_with_same_semantics(tmp_path: Path) -> None:
    registered_root, test_root = create_roots(tmp_path / "dataset")
    json_path = test_root / "annotations.json"
    jsonl_path = test_root / "annotations.jsonl"

    payload = {
        "group_01.jpg": [
            {"bbox": [0, 0, 8, 8], "identity": "p03"},
            {"bbox": [1, 1, 6, 6], "identity": "unknown"},
        ],
        "p03_t01.jpg": [
            {"bbox": [0, 0, 8, 8], "identity": "p03"},
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image": "group_01.jpg",
                        "faces": [
                            {"bbox": [0, 0, 8, 8], "identity": "p03"},
                            {"bbox": [1, 1, 6, 6], "identity": "unknown"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "image": "p03_t01.jpg",
                        "faces": [{"bbox": [0, 0, 8, 8], "identity": "p03"}],
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    from_json = load_grouped_self_dataset(json_path, test_root, registered_root)
    from_jsonl = load_grouped_self_dataset(jsonl_path, test_root, registered_root)

    assert from_json == from_jsonl


def test_load_grouped_self_dataset_rejects_invalid_bbox_with_readable_error(tmp_path: Path) -> None:
    registered_root, test_root = create_roots(tmp_path / "dataset")
    annotations_path = test_root / "annotations.json"
    annotations_path.write_text(
        json.dumps({"group_01.jpg": [{"bbox": [0, 0, 8], "identity": "p03"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="group_01.jpg"):
        load_grouped_self_dataset(annotations_path, test_root, registered_root)


def test_load_grouped_self_dataset_rejects_unknown_identity_value(tmp_path: Path) -> None:
    registered_root, test_root = create_roots(tmp_path / "dataset")
    annotations_path = test_root / "annotations.jsonl"
    annotations_path.write_text(
        json.dumps(
            {
                "image": "group_01.jpg",
                "faces": [{"bbox": [0, 0, 8, 8], "identity": "alice"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="alice"):
        load_grouped_self_dataset(annotations_path, test_root, registered_root)


def test_load_grouped_self_dataset_rejects_missing_referenced_image(tmp_path: Path) -> None:
    registered_root, test_root = create_roots(tmp_path / "dataset")
    annotations_path = test_root / "annotations.json"
    annotations_path.write_text(
        json.dumps({"missing.jpg": [{"bbox": [0, 0, 8, 8], "identity": "p03"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        load_grouped_self_dataset(annotations_path, test_root, registered_root)
