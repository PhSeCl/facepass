import cv2
import numpy as np

from src.common.errors import InvalidImageError
from src.common.logging import get_logger

from .base import FaceModel
from .schemas import DetectedFace


logger = get_logger(__name__)


def _ensure_valid_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("图片必须是 numpy.ndarray")
    if image.size == 0 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise InvalidImageError("图片数组为空或维度不正确")


def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise InvalidImageError("模型返回了零向量 embedding")
    return vector / norm


class InsightFaceModel(FaceModel):
    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name=model_name,
                providers=providers or ["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=det_size)
            self.recognition_model = next(
                model for model in self.app.models.values() if hasattr(model, "get_feat")
            )
        except Exception as exc:
            logger.error("模型加载失败: %s", exc)
            raise

    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        _ensure_valid_image(image)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        faces = self.app.get(image)
        if not faces:
            logger.info("未检测到人脸")
            return []

        detected: list[DetectedFace] = []
        for face in faces:
            x1, y1, x2, y2 = [int(round(value)) for value in face.bbox]
            embedding = _l2_normalize(face.embedding)
            landmarks = getattr(face, "kps", None)
            detected.append(
                DetectedFace(
                    bbox=(x1, y1, max(0, x2 - x1), max(0, y2 - y1)),
                    embedding=embedding,
                    det_score=float(face.det_score),
                    landmarks=np.asarray(landmarks, dtype=np.float32) if landmarks is not None else None,
                )
            )
        return detected

    def encode_aligned(self, face_image: np.ndarray) -> np.ndarray:
        _ensure_valid_image(face_image)
        return _l2_normalize(self.recognition_model.get_feat(face_image).flatten())
