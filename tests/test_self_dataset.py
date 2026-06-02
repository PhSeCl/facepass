import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from src.eval.self_dataset import load_self_dataset
from scripts.eval_self import build_parser, main


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

    group = np.zeros((8, 16, 3), dtype=np.uint8)
    group[:, :8] = (255, 0, 0)
    group[:, 8:] = (0, 0, 255)
    write_image(images_root / "group.png", group)

    single = np.zeros((8, 8, 3), dtype=np.uint8)
    single[:, :] = (0, 255, 0)
    write_image(images_root / "single.png", single)

    records = [
        {
            "image_path": "images/group.png",
            "faces": [
                {"identity_id": "p01", "bbox": [0, 0, 8, 8]},
                {"identity_id": "unknown", "bbox": [8, 0, 8, 8]},
            ],
        },
        {
            "image_path": "images/single.png",
            "faces": [
                {"identity_id": "p02", "bbox": [0, 0, 8, 8]},
            ],
        },
    ]
    annotations_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    return registered_root, test_root, annotations_path


def test_load_self_dataset_parses_annotations_and_crops_faces(tmp_path) -> None:
    registered_root, test_root, annotations_path = create_fixture(tmp_path / "self_eval")

    dataset = load_self_dataset(annotations_path, test_root)

    assert dataset.registered_root == registered_root
    assert dataset.test_root == test_root
    assert len(dataset.samples) == 3
    assert [sample.identity_id for sample in dataset.samples] == ["p01", "unknown", "p02"]
    assert dataset.samples[0].bbox == (0, 0, 8, 8)
    assert dataset.samples[1].bbox == (8, 0, 8, 8)
    assert dataset.samples[0].face_image.shape == (8, 8, 3)
    assert dataset.samples[1].face_image.shape == (8, 8, 3)
    assert float(dataset.samples[0].face_image[:, :, 2].mean()) > float(
        dataset.samples[0].face_image[:, :, 1].mean()
    )
    assert float(dataset.samples[1].face_image[:, :, 0].mean()) > float(
        dataset.samples[1].face_image[:, :, 2].mean()
    )


def test_eval_self_returns_friendly_error_when_annotations_missing(tmp_path, capsys) -> None:
    report_path = tmp_path / "reports" / "self_eval.json"

    exit_code = main(
        [
            "--annotations-path",
            str(tmp_path / "missing" / "annotations.jsonl"),
            "--test-root",
            str(tmp_path / "missing"),
            "--registered-root",
            str(tmp_path / "registered"),
            "--report-path",
            str(report_path),
            "--model-name",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "不存在" in captured.out
    assert not report_path.exists()


def test_eval_self_parser_defaults_to_dataset_directories() -> None:
    args = build_parser().parse_args([])

    assert args.annotations_path == "dataset/test/annotations.jsonl"
    assert args.test_root == "dataset/test"
    assert args.registered_root == "dataset/registered"


def test_eval_self_script_runs_with_fake_model_and_writes_report(tmp_path) -> None:
    registered_root, test_root, annotations_path = create_fixture(tmp_path / "self_eval")
    report_path = tmp_path / "reports" / "self_eval.json"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_self.py",
            "--annotations-path",
            str(annotations_path),
            "--test-root",
            str(test_root),
            "--registered-root",
            str(registered_root),
            "--report-path",
            str(report_path),
            "--model-name",
            "fake",
            "--threshold",
            "0.1",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert "known top-1 accuracy" in result.stdout
    assert report["known_top1_accuracy"] == 1.0
    assert report["known_total"] == 2
    assert report["known_correct"] == 2
    assert report["unknown_total"] == 1
    assert report["unknown_predicted_unknown"] == 1
    assert report["unknown_accuracy"] == 1.0
    assert len(report["samples"]) == 3
