import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from scripts.eval_end2end import main


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
        images_root / "p01.png",
        np.full((8, 8, 3), fill_value=(255, 0, 0), dtype=np.uint8),
    )
    write_image(
        images_root / "p02.png",
        np.full((8, 8, 3), fill_value=(0, 255, 0), dtype=np.uint8),
    )
    write_image(
        images_root / "unknown.png",
        np.full((8, 8, 3), fill_value=(0, 0, 255), dtype=np.uint8),
    )

    records = [
        {
            "image_path": "images/p01.png",
            "faces": [{"identity_id": "p01", "bbox": [0, 0, 8, 8]}],
        },
        {
            "image_path": "images/p02.png",
            "faces": [{"identity_id": "p02", "bbox": [0, 0, 8, 8]}],
        },
        {
            "image_path": "images/unknown.png",
            "faces": [{"identity_id": "unknown", "bbox": [0, 0, 8, 8]}],
        },
    ]
    annotations_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    return registered_root, test_root, annotations_path


def test_eval_end2end_returns_friendly_message_when_dataset_is_missing(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--annotations-path",
            str(tmp_path / "missing" / "annotations.jsonl"),
            "--test-root",
            str(tmp_path / "missing"),
            "--registered-root",
            str(tmp_path / "registered"),
            "--model-name",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "需要 data/test 标注" in captured.out


def test_eval_end2end_script_writes_report_and_plots(tmp_path) -> None:
    registered_root, test_root, annotations_path = create_fixture(tmp_path / "end2end")
    report_path = tmp_path / "reports" / "end2end_eval.json"
    confusion_path = tmp_path / "reports" / "end2end_confusion_matrix.png"
    detection_path = tmp_path / "reports" / "end2end_detection.png"
    accuracy_path = tmp_path / "reports" / "end2end_accuracy.png"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_end2end.py",
            "--annotations-path",
            str(annotations_path),
            "--test-root",
            str(test_root),
            "--registered-root",
            str(registered_root),
            "--report-path",
            str(report_path),
            "--confusion-matrix-path",
            str(confusion_path),
            "--detection-plot-path",
            str(detection_path),
            "--accuracy-plot-path",
            str(accuracy_path),
            "--model-name",
            "fake",
            "--threshold",
            "0.8",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert "strict top-1 accuracy" in result.stdout
    assert report["ground_truth_total"] == 3
    assert report["matched_ground_truth_total"] == 3
    assert report["detection_recall"] == 1.0
    assert report["strict_top1_accuracy"] == 1.0
    assert report["matched_top1_accuracy"] == 1.0
    assert report["unknown_detected_accuracy"] == 1.0
    assert report["predicted_unknown_precision"] == 1.0
    assert confusion_path.exists()
    assert detection_path.exists()
    assert accuracy_path.exists()
