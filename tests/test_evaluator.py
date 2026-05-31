import numpy as np
import pytest

from src.backend.gallery import Gallery
from src.eval.evaluator import EvalReport, EvalSample, evaluate


def unit(values: list[float]) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class StubModel:
    def __init__(self, mapping: dict[tuple[int, int, int], np.ndarray]) -> None:
        self.mapping = mapping

    def encode_aligned(self, face_image: np.ndarray) -> np.ndarray:
        key = tuple(int(value) for value in face_image[0, 0])
        return self.mapping[key]


def image_with_key(key: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :] = key
    return image


def test_evaluate_reports_perfect_closed_set_accuracy() -> None:
    gallery = Gallery()
    gallery.register("p01", [unit([1, 0, 0])])
    gallery.register("p02", [unit([0, 1, 0])])
    model = StubModel(
        {
            (255, 0, 0): unit([1, 0, 0]),
            (0, 255, 0): unit([0, 1, 0]),
        }
    )
    samples = [
        (image_with_key((255, 0, 0)), "p01"),
        (image_with_key((0, 255, 0)), "p02"),
    ]

    report = evaluate(model, gallery, samples)

    assert isinstance(report, EvalReport)
    assert report.metrics.top1_accuracy == 1.0
    assert report.metrics.per_class["p01"].accuracy == 1.0
    assert report.metrics.per_class["p02"].accuracy == 1.0
    assert report.samples == [
        EvalSample(true_identity_id="p01", predicted_identity_id="p01", similarity=1.0),
        EvalSample(true_identity_id="p02", predicted_identity_id="p02", similarity=1.0),
    ]


def test_evaluate_applies_threshold_without_importing_recognizer() -> None:
    gallery = Gallery()
    gallery.register("p01", [unit([1, 0, 0])])
    gallery.register("p02", [unit([0, 1, 0])])
    model = StubModel({(1, 1, 1): unit([0.8, 0.6, 0])})
    samples = [(image_with_key((1, 1, 1)), "p01")]

    report = evaluate(model, gallery, samples, threshold=0.9)

    assert report.metrics.top1_accuracy == 0.0
    assert len(report.samples) == 1
    assert report.samples[0].true_identity_id == "p01"
    assert report.samples[0].predicted_identity_id == "unknown"
    assert report.samples[0].similarity == pytest.approx(0.8)


def test_evaluate_marks_unknown_when_gallery_has_no_candidates() -> None:
    model = StubModel({(9, 9, 9): unit([1, 0, 0])})
    samples = [(image_with_key((9, 9, 9)), "p01")]

    report = evaluate(model, Gallery(), samples)

    assert report.metrics.top1_accuracy == 0.0
    assert report.samples == [
        EvalSample(true_identity_id="p01", predicted_identity_id="unknown", similarity=0.0)
    ]
