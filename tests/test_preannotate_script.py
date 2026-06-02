import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.backend.gallery import Gallery
from src.backend.recognizer import Recognizer
from src.eval.end2end_dataset import load_grouped_self_dataset
from src.face_model.schemas import DetectedFace
from scripts.preannotate_test import build_parser, main
from src.eval.preannotation import generate_draft_annotations


def unit(values: list[float]) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def write_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


class ScenarioModel:
    def encode_aligned(self, face_image: np.ndarray) -> np.ndarray:
        blue, green, red = (int(value) for value in face_image[0, 0, :3])
        if blue < 20 and green < 20 and red > 240:
            return unit([1, 0, 0])
        if blue < 20 and green > 240 and red < 20:
            return unit([0, 1, 0])
        raise AssertionError(f"unexpected aligned face signature: {(blue, green, red)}")

    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        blue, green, red = (int(value) for value in image[0, 0, :3])
        if blue < 20 and green > 240 and red > 240:
            return [
                DetectedFace(
                    bbox=(1, 2, 3, 4),
                    embedding=unit([1, 0, 0]),
                    det_score=1.0,
                    landmarks=None,
                )
            ]
        if blue > 240 and green > 240 and red < 20:
            return [
                DetectedFace(
                    bbox=(5, 6, 7, 8),
                    embedding=unit([0.2, 0.34, 0.9193476]),
                    det_score=0.8,
                    landmarks=None,
                )
            ]
        if blue > 240 and green < 20 and red < 20:
            return [
                DetectedFace(
                    bbox=(9, 10, 11, 12),
                    embedding=unit([0.2, 0.1, 0.9746794]),
                    det_score=0.8,
                    landmarks=None,
                )
            ]
        if blue > 240 and green < 20 and red > 240:
            return [
                DetectedFace(
                    bbox=(0, 0, 10, 10),
                    embedding=unit([1, 0, 0]),
                    det_score=0.9,
                    landmarks=None,
                ),
                DetectedFace(
                    bbox=(10, 0, 10, 10),
                    embedding=unit([0, 1, 0]),
                    det_score=0.9,
                    landmarks=None,
                ),
            ]
        if blue < 20 and green > 200 and red > 100:
            return [
                DetectedFace(
                    bbox=(20, 20, 30, 30),
                    embedding=unit([0.95, 0.3122499, 0.0]),
                    det_score=0.9,
                    landmarks=None,
                ),
                DetectedFace(
                    bbox=(60, 20, 30, 30),
                    embedding=unit([0.80, 0.60, 0.0]),
                    det_score=0.85,
                    landmarks=None,
                ),
            ]
        if blue == green == red:
            return []
        raise AssertionError(f"unexpected image signature: {(blue, green, red)}")


def test_generate_draft_annotations_writes_scores_and_review_warnings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_root = tmp_path / "dataset" / "test"
    images_dir = test_root / "images"
    registered_root = tmp_path / "dataset" / "registered"
    output_path = test_root / "annotation.json"

    write_image(registered_root / "p01" / "r1.jpg", np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8))
    write_image(registered_root / "p02" / "r1.jpg", np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8))
    write_image(images_dir / "p01_t01.jpg", np.full((8, 8, 3), (255, 255, 0), dtype=np.uint8))
    write_image(images_dir / "p02_t02.jpg", np.full((8, 8, 3), (0, 255, 255), dtype=np.uint8))
    write_image(images_dir / "p04_t04.jpg", np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8))
    write_image(images_dir / "p03_t03.jpg", np.full((8, 8, 3), (255, 0, 255), dtype=np.uint8))
    write_image(images_dir / "group_dup.jpg", np.full((8, 8, 3), (120, 255, 0), dtype=np.uint8))
    write_image(images_dir / "group_01.jpg", np.full((8, 8, 3), (128, 128, 128), dtype=np.uint8))

    gallery = Gallery()
    gallery.register("p01", [unit([1, 0, 0])])
    gallery.register("p02", [unit([0, 1, 0])])
    recognizer = Recognizer(
        model=ScenarioModel(),
        gallery=gallery,
        threshold=0.8,
        id2name={"p01": "Alice", "p02": "Bob"},
    )

    with caplog.at_level("WARNING"):
        summary = generate_draft_annotations(
            recognizer=recognizer,
            images_dir=images_dir,
            out_path=output_path,
            review_threshold=0.45,
            draft_identity_threshold=0.25,
            overwrite=False,
        )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    dataset = load_grouped_self_dataset(output_path, test_root, registered_root)

    assert payload["p01_t01.jpg"] == [{"bbox": [1, 2, 3, 4], "identity": "p01", "score": 1.0}]
    assert payload["p02_t02.jpg"][0]["identity"] == "p02"
    assert payload["p02_t02.jpg"][0]["score"] == pytest.approx(0.34, abs=1e-3)
    assert payload["p04_t04.jpg"] == [{"bbox": [9, 10, 11, 12], "identity": "unknown", "score": pytest.approx(0.2, abs=1e-3)}]
    assert payload["p03_t03.jpg"][0]["identity"] == "p01"
    assert payload["p03_t03.jpg"][1]["identity"] == "p02"
    assert payload["group_dup.jpg"] == [
        {"bbox": [20, 20, 30, 30], "identity": "p01", "score": pytest.approx(0.95, abs=1e-3)},
        {"bbox": [60, 20, 30, 30], "identity": "unknown", "score": pytest.approx(0.8, abs=1e-3)},
    ]
    assert payload["group_01.jpg"] == []
    assert [item.image_path.name for item in dataset.images] == [
        "group_01.jpg",
        "group_dup.jpg",
        "p01_t01.jpg",
        "p02_t02.jpg",
        "p03_t03.jpg",
        "p04_t04.jpg",
    ]
    assert summary.processed_images == 6
    assert summary.total_faces == 7
    assert summary.review_images == 5
    assert "p02_t02.jpg" in caplog.text
    assert "p04_t04.jpg" in caplog.text
    assert "低置信度" in caplog.text
    assert "p03_t03.jpg" in caplog.text
    assert "单人照检出多脸" in caplog.text
    assert "group_dup.jpg" in caplog.text
    assert "同图存在相似人物" in caplog.text
    assert "group_01.jpg" in caplog.text
    assert "未检出人脸" in caplog.text


def test_preannotate_parser_defaults_to_dataset_directories() -> None:
    args = build_parser().parse_args([])

    assert args.images_dir == "dataset/test/images"
    assert args.out == "dataset/test/annotation.json"
    assert args.registered_root == "dataset/registered"
    assert args.review_threshold == 0.45
    assert args.draft_identity_threshold == 0.25
    assert args.overwrite is False


def test_preannotate_main_refuses_to_overwrite_existing_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    images_dir = tmp_path / "dataset" / "test" / "images"
    images_dir.mkdir(parents=True)
    output_path = tmp_path / "dataset" / "test" / "annotation.json"
    output_path.write_text('{"existing": []}', encoding="utf-8")

    import scripts.preannotate_test as script_module

    monkeypatch.setattr(
        script_module,
        "build_recognizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("build_recognizer should not run")),
    )

    exit_code = main(
        [
            "--images-dir",
            str(images_dir),
            "--out",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "已存在" in captured.out
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"existing": []}


def test_preannotate_main_runs_with_fake_model_and_writes_annotation(tmp_path: Path) -> None:
    registered_root = tmp_path / "dataset" / "registered"
    images_dir = tmp_path / "dataset" / "test" / "images"
    output_path = tmp_path / "dataset" / "test" / "annotation.json"
    gallery_path = tmp_path / "models" / "gallery.pkl"

    write_image(registered_root / "p01" / "r1.jpg", np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8))
    write_image(registered_root / "p02" / "r1.jpg", np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8))
    write_image(images_dir / "p02_t01.jpg", np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8))

    exit_code = main(
        [
            "--images-dir",
            str(images_dir),
            "--out",
            str(output_path),
            "--registered-root",
            str(registered_root),
            "--gallery-path",
            str(gallery_path),
            "--model-name",
            "fake",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    dataset = load_grouped_self_dataset(output_path, images_dir.parent, registered_root)

    assert exit_code == 0
    assert gallery_path.exists()
    assert payload["p02_t01.jpg"][0]["identity"] == "p02"
    assert payload["p02_t01.jpg"][0]["score"] == 1.0
    assert len(dataset.images) == 1
    assert dataset.images[0].image_path.name == "p02_t01.jpg"
