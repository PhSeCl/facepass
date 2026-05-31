import numpy as np

from src.common.errors import InvalidImageError

from .base import FaceModel
from .schemas import DetectedFace


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("图片必须是 numpy.ndarray")
    if image.size == 0 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise InvalidImageError("图片数组为空或维度不正确")
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise InvalidImageError("fake model 产生了零向量")
    return array / norm


class FakeFaceModel(FaceModel):
    """Deterministic test double that never downloads weights."""

    def _embedding_for(self, image: np.ndarray) -> np.ndarray:
        image = _validate_image(image)
        mean_rgb = image[:, :, :3].mean(axis=(0, 1))
        dominant_channel = int(np.argmax(mean_rgb))
        basis = np.zeros(3, dtype=np.float32)
        basis[dominant_channel] = 1.0
        return _unit(basis)

    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        validated = _validate_image(image)
        height, width = validated.shape[:2]
        return [
            DetectedFace(
                bbox=(0, 0, int(width), int(height)),
                embedding=self._embedding_for(validated),
                det_score=1.0,
                landmarks=None,
            )
        ]

    def encode_aligned(self, face_image: np.ndarray) -> np.ndarray:
        return self._embedding_for(face_image)
