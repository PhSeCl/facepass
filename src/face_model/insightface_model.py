import importlib
from pathlib import Path

import cv2
import numpy as np

from src.common.errors import InvalidImageError, ModelConfigError, ModelLoadError
from src.common.logging import get_logger

from .base import FaceModel
from . import model_config
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


def _create_face_analysis_app(
    model_path: Path,
    model_name: str,
    providers: list[str] | None,
):
    face_analysis_module = importlib.import_module("insightface.app.face_analysis")
    original_ensure_available = face_analysis_module.ensure_available
    face_analysis_module.ensure_available = lambda sub_dir, name, root="~/.insightface": str(model_path)
    try:
        return face_analysis_module.FaceAnalysis(
            name=model_name,
            root=str(model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
    finally:
        face_analysis_module.ensure_available = original_ensure_available


def _wrap_model_runtime_error(exc: Exception, *, action: str, model_path: Path | None = None) -> ModelLoadError:
    if isinstance(exc, ModelLoadError):
        return exc

    message = f"模型{action}失败: {exc}"
    if model_path is not None:
        message = (
            f"{message}。当前使用的持久化/默认模型路径可能已失效，请检查 config.toml，"
            "或通过 --model-path 重新指定有效模型目录。"
        )
    return ModelLoadError(message)


class InsightFaceModel(FaceModel):
    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
        model_path: str | Path | None = None,
        gui_model_path: str | Path | None = None,
    ) -> None:
        explicit_model_path = model_path is not None or gui_model_path is not None
        validated_model_path: Path | None = None
        try:
            resolved_model_path = model_config.resolve_model_path(
                cli_path=model_path,
                gui_path=gui_model_path,
            )
            validated_model_path = model_config.validate_model_path(resolved_model_path)

            self.app = _create_face_analysis_app(
                model_path=validated_model_path,
                model_name=model_name,
                providers=providers,
            )
            self.app.prepare(ctx_id=0, det_size=det_size)
            self.recognition_model = next(
                model for model in self.app.models.values() if hasattr(model, "get_feat")
            )
            if explicit_model_path:
                model_config.persist_path(validated_model_path)
        except ModelConfigError as exc:
            logger.error("模型加载失败: %s", exc)
            raise
        except Exception as exc:
            wrapped = _wrap_model_runtime_error(
                exc,
                action="加载",
                model_path=None if explicit_model_path else validated_model_path,
            )
            logger.error("%s", wrapped)
            raise wrapped from exc

    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        _ensure_valid_image(image)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        try:
            faces = self.app.get(image)
        except Exception as exc:
            wrapped = _wrap_model_runtime_error(exc, action="推理")
            logger.error("%s", wrapped)
            raise wrapped from exc
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
        try:
            embedding = self.recognition_model.get_feat(face_image).flatten()
        except Exception as exc:
            wrapped = _wrap_model_runtime_error(exc, action="推理")
            logger.error("%s", wrapped)
            raise wrapped from exc
        return _l2_normalize(embedding)
