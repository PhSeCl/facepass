import numpy as np

from src.face_model.insightface_model import InsightFaceModel


class FakeRecognitionModel:
    def get_feat(self, image: np.ndarray) -> np.ndarray:
        return np.array([[3.0, 4.0]], dtype=np.float32)


def test_encode_aligned_uses_recognition_model_without_detection() -> None:
    model = object.__new__(InsightFaceModel)
    model.recognition_model = FakeRecognitionModel()

    embedding = model.encode_aligned(np.zeros((8, 8, 3), dtype=np.uint8))

    assert np.allclose(embedding, np.array([0.6, 0.8], dtype=np.float32))
