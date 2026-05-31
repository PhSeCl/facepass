import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.end2end_dataset import load_grouped_self_dataset
from src.eval.end2end_evaluator import evaluate_end2end
from src.face_model.fake_model import FakeFaceModel
from src.face_model.schemas import DetectedFace


def unit(values: list[float]) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def write_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def create_fixture(root: Path) -> tuple[Path, Path, Path]:
    registered_root = root / "registered"
    test_root = root / "test"
    images_root = test_root / "images"
    annotations_path = test_root / "annotations.jsonl"

    write_image(
        registered_root / "p01" / "r1.png",
        np.full((8, 8, 3), fill_value=(255, 0, 0), dtype=np.uint8),
    )
    write_image(
        registered_root / "p02" / "r1.png",
        np.full((8, 8, 3), fill_value=(0, 255, 0), dtype=np.uint8),
    )

    write_image(
        images_root / "perfect.png",
        np.full((8, 8, 3), fill_value=(255, 255, 0), dtype=np.uint8),
    )
    write_image(
        images_root / "unknown.png",
        np.full((8, 8, 3), fill_value=(0, 255, 255), dtype=np.uint8),
    )
    write_image(
        images_root / "miss.png",
        np.full((8, 8, 3), fill_value=(255, 0, 255), dtype=np.uint8),
    )
    false_positive = np.zeros((8, 24, 3), dtype=np.uint8)
    false_positive[:, :] = (128, 128, 128)
    write_image(images_root / "fp.png", false_positive)

    records = [
        {
            "image_path": "images/perfect.png",
            "faces": [{"identity_id": "p01", "bbox": [0, 0, 8, 8]}],
        },
        {
            "image_path": "images/unknown.png",
            "faces": [{"identity_id": "unknown", "bbox": [0, 0, 8, 8]}],
        },
        {
            "image_path": "images/miss.png",
            "faces": [{"identity_id": "p02", "bbox": [0, 0, 8, 8]}],
        },
        {
            "image_path": "images/fp.png",
            "faces": [{"identity_id": "p02", "bbox": [0, 0, 8, 8]}],
        },
    ]
    annotations_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    return registered_root, test_root, annotations_path


class ScenarioFaceModel(FakeFaceModel):
    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        signature = tuple(int(value) for value in image[0, 0, :3])
        if signature == (0, 0, 255):  # red registration image after BGR conversion
            return [
                DetectedFace(
                    bbox=(0, 0, 8, 8),
                    embedding=unit([1, 0, 0]),
                    det_score=1.0,
                    landmarks=None,
                )
            ]
        if signature == (0, 255, 0):  # green registration image
            return [
                DetectedFace(
                    bbox=(0, 0, 8, 8),
                    embedding=unit([0, 1, 0]),
                    det_score=1.0,
                    landmarks=None,
                )
            ]
        if signature == (0, 255, 255):  # perfect known-face image
            return [
                DetectedFace(
                    bbox=(0, 0, 8, 8),
                    embedding=unit([1, 0, 0]),
                    det_score=1.0,
                    landmarks=None,
                )
            ]
        if signature == (255, 255, 0):  # unknown face image
            return [
                DetectedFace(
                    bbox=(0, 0, 8, 8),
                    embedding=unit([1, 1, 1]),
                    det_score=1.0,
                    landmarks=None,
                )
            ]
        if signature == (255, 0, 255):  # missed detection image
            return []
        if signature == (128, 128, 128):  # wrong identity + extra false positive
            return [
                DetectedFace(
                    bbox=(0, 0, 8, 8),
                    embedding=unit([1, 0, 0]),
                    det_score=1.0,
                    landmarks=None,
                ),
                DetectedFace(
                    bbox=(16, 0, 8, 8),
                    embedding=unit([0, 1, 0]),
                    det_score=0.9,
                    landmarks=None,
                ),
            ]
        raise AssertionError(f"unexpected test image signature: {signature}")


def test_load_grouped_self_dataset_groups_faces_by_image(tmp_path) -> None:
    registered_root, test_root, annotations_path = create_fixture(tmp_path / "end2end_eval")

    dataset = load_grouped_self_dataset(annotations_path, test_root)

    assert dataset.registered_root == registered_root
    assert dataset.test_root == test_root
    assert [sample.image_path.name for sample in dataset.images] == [
        "perfect.png",
        "unknown.png",
        "miss.png",
        "fp.png",
    ]
    assert [(face.identity_id, face.bbox) for face in dataset.images[0].faces] == [
        ("p01", (0, 0, 8, 8))
    ]
    assert [(face.identity_id, face.bbox) for face in dataset.images[1].faces] == [
        ("unknown", (0, 0, 8, 8))
    ]


def test_evaluate_end2end_reports_metrics_and_logs_misses_and_false_positives(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, test_root, annotations_path = create_fixture(tmp_path / "end2end_eval")
    dataset = load_grouped_self_dataset(annotations_path, test_root)

    with caplog.at_level("WARNING"):
        report = evaluate_end2end(
            dataset=dataset,
            model=ScenarioFaceModel(),
            threshold=0.8,
        )

    assert len(report.image_results) == 4
    assert report.metrics.ground_truth_total == 4
    assert report.metrics.matched_ground_truth_total == 3
    assert report.metrics.detection_recall == 0.75
    assert report.metrics.false_positives == 1
    assert report.metrics.strict_top1_correct == 2
    assert report.metrics.strict_top1_accuracy == 0.5
    assert report.metrics.matched_top1_correct == 2
    assert report.metrics.matched_top1_accuracy == pytest.approx(2 / 3)
    assert report.metrics.unknown_detected_total == 1
    assert report.metrics.unknown_detected_correct == 1
    assert report.metrics.unknown_detected_accuracy == 1.0
    assert report.metrics.predicted_unknown_total == 1
    assert report.metrics.predicted_unknown_correct == 1
    assert report.metrics.predicted_unknown_precision == 1.0
    assert report.metrics.confusion_pairs == [
        ("p01", "p01"),
        ("unknown", "unknown"),
        ("p02", "p01"),
    ]
    assert "missed detection" in caplog.text
    assert "miss.png" in caplog.text
    assert "false positive" in caplog.text
    assert "fp.png" in caplog.text
