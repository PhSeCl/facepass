import numpy as np

from src.backend.gallery import Gallery
from src.backend.recognizer import Recognizer
from src.face_model.schemas import DetectedFace


class FakeModel:
    def __init__(self, embedding: np.ndarray) -> None:
        self.embedding = embedding

    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        return [
            DetectedFace(
                bbox=(10, 20, 30, 40),
                embedding=self.embedding,
                det_score=0.9,
                landmarks=None,
            )
        ]

    def encode_aligned(self, face_image: np.ndarray) -> np.ndarray:
        return self.embedding


def unit(values: list[float]) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_recognizer_marks_match_when_similarity_meets_threshold() -> None:
    gallery = Gallery()
    gallery.register("p01", [unit([1, 0, 0])])
    recognizer = Recognizer(
        model=FakeModel(unit([1, 0, 0])),
        gallery=gallery,
        threshold=0.3,
        id2name={"p01": "Alice"},
    )

    results = recognizer.recognize_image(np.zeros((8, 8, 3), dtype=np.uint8))

    assert len(results) == 1
    assert results[0].identity_id == "p01"
    assert results[0].name == "Alice"
    assert results[0].is_unknown is False
    assert results[0].similarity == 1.0


def test_recognizer_marks_unknown_below_threshold() -> None:
    gallery = Gallery()
    gallery.register("p01", [unit([1, 0, 0])])
    recognizer = Recognizer(
        model=FakeModel(unit([0, 1, 0])),
        gallery=gallery,
        threshold=0.3,
        id2name={"p01": "Alice"},
    )

    results = recognizer.recognize_image(np.zeros((8, 8, 3), dtype=np.uint8))

    assert results[0].identity_id == "unknown"
    assert results[0].name is None
    assert results[0].is_unknown is True
    assert results[0].similarity == 0.0


def test_recognizer_marks_unknown_when_gallery_has_no_candidates() -> None:
    recognizer = Recognizer(
        model=FakeModel(unit([1, 0, 0])),
        gallery=Gallery(),
        threshold=0.3,
        id2name={"p01": "Alice"},
    )

    results = recognizer.recognize_image(np.zeros((8, 8, 3), dtype=np.uint8))

    assert results[0].identity_id == "unknown"
    assert results[0].name is None
    assert results[0].is_unknown is True
    assert results[0].similarity == 0.0
