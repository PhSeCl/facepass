import importlib
import os
from pathlib import Path

import cv2
import numpy as np

from src.common.errors import InvalidImageError, ModelConfigError, ModelLoadError
from src.common.logging import get_logger

from .base import FaceModel
from . import model_config
from .schemas import DetectedFace


logger = get_logger(__name__)
_DLL_DIRECTORY_HANDLES: list[object] = []
_REGISTERED_DLL_DIRECTORIES: set[str] = set()


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


def _default_execution_providers() -> list[str]:
    try:
        onnxruntime_module = importlib.import_module("onnxruntime")
        available_providers = onnxruntime_module.get_available_providers()
    except Exception:
        logger.warning("无法检测 onnxruntime provider，回退到 CPUExecutionProvider")
        return ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_runtime_diagnostics() -> dict[str, object]:
    try:
        onnxruntime_module = importlib.import_module("onnxruntime")
    except Exception as exc:
        return {
            "onnxruntime_version": None,
            "device": None,
            "available_providers": [],
            "preferred_providers": ["CPUExecutionProvider"],
            "gpu_enabled": False,
            "error": str(exc),
        }

    available_providers = list(onnxruntime_module.get_available_providers())
    preferred_providers = _default_execution_providers()
    device = onnxruntime_module.get_device() if hasattr(onnxruntime_module, "get_device") else None
    return {
        "onnxruntime_version": getattr(onnxruntime_module, "__version__", None),
        "device": device,
        "available_providers": available_providers,
        "preferred_providers": preferred_providers,
        "gpu_enabled": "CUDAExecutionProvider" in preferred_providers,
    }


def _register_nvidia_dll_directories(onnxruntime_module) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    module_file = getattr(onnxruntime_module, "__file__", None)
    if module_file is None:
        return

    site_packages = Path(module_file).resolve().parent.parent
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.is_dir():
        return

    existing_path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    for bin_dir in sorted(nvidia_root.glob("*/bin")):
        if not bin_dir.is_dir():
            continue

        normalized = str(bin_dir)
        if normalized not in _REGISTERED_DLL_DIRECTORIES:
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(normalized))
                _REGISTERED_DLL_DIRECTORIES.add(normalized)
            except OSError as exc:
                logger.warning("注册 NVIDIA DLL 目录失败 %s: %s", normalized, exc)
                continue

        if normalized not in existing_path_parts:
            existing_path_parts.insert(0, normalized)

    if existing_path_parts:
        os.environ["PATH"] = os.pathsep.join(existing_path_parts)


def _maybe_preload_gpu_dlls(providers: list[str]) -> None:
    if "CUDAExecutionProvider" not in providers:
        return

    try:
        onnxruntime_module = importlib.import_module("onnxruntime")
    except Exception as exc:
        logger.warning("CUDA provider 已选中，但导入 onnxruntime 失败，跳过 GPU DLL 预加载: %s", exc)
        return

    preload_dlls = getattr(onnxruntime_module, "preload_dlls", None)
    if preload_dlls is None:
        logger.warning("当前 onnxruntime 不支持 preload_dlls，跳过 GPU DLL 预加载")
        return

    try:
        _register_nvidia_dll_directories(onnxruntime_module)
        # Empty string tells onnxruntime to search NVIDIA site-packages first,
        # then fall back to its default DLL lookup order.
        preload_dlls(directory="")
    except Exception as exc:
        logger.warning("GPU DLL 预加载失败，后续将由 onnxruntime 自行解析 provider: %s", exc)


def _create_face_analysis_app(
    model_path: Path,
    model_name: str,
    providers: list[str],
):
    _maybe_preload_gpu_dlls(providers)
    face_analysis_module = importlib.import_module("insightface.app.face_analysis")
    original_ensure_available = face_analysis_module.ensure_available
    face_analysis_module.ensure_available = lambda sub_dir, name, root="~/.insightface": str(model_path)
    try:
        return face_analysis_module.FaceAnalysis(
            name=model_name,
            root=str(model_path),
            providers=providers,
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


def _is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).upper()
    markers = (
        "CUDA",
        "CUDNN",
        "EP_FAIL",
        "CUDAEXECUTIONPROVIDER",
        "TENSORRT",
    )
    return any(marker in message for marker in markers)


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
        resolved_providers = providers or _default_execution_providers()
        self.model_name = model_name
        self.det_size = det_size
        self.model_path: Path | None = None
        self.providers = list(resolved_providers)
        self._cpu_fallback_used = False
        try:
            resolved_model_path = model_config.resolve_model_path(
                cli_path=model_path,
                gui_path=gui_model_path,
            )
            validated_model_path = model_config.validate_model_path(resolved_model_path)
            self.model_path = validated_model_path

            self._load_runtime(self.providers)
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

    def _load_runtime(self, providers: list[str]) -> None:
        if self.model_path is None:
            raise ModelLoadError("模型路径未初始化，无法装载推理运行时")

        self.app = _create_face_analysis_app(
            model_path=self.model_path,
            model_name=self.model_name,
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=self.det_size)
        self.recognition_model = next(
            model for model in self.app.models.values() if hasattr(model, "get_feat")
        )
        self.providers = list(providers)

    def _retry_with_cpu_fallback(self, exc: Exception) -> bool:
        if getattr(self, "_cpu_fallback_used", False):
            return False
        if not _is_cuda_runtime_error(exc):
            return False
        providers = getattr(self, "providers", None)
        if not providers:
            return False
        if providers == ["CPUExecutionProvider"]:
            return False
        if "CPUExecutionProvider" not in providers:
            return False

        logger.warning(
            "GPU 推理失败，尝试回退到 CPUExecutionProvider 重新加载 InsightFace runtime: %s",
            exc,
        )
        self._load_runtime(["CPUExecutionProvider"])
        self._cpu_fallback_used = True
        return True

    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        _ensure_valid_image(image)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        try:
            faces = self.app.get(image)
        except Exception as exc:
            if self._retry_with_cpu_fallback(exc):
                faces = self.app.get(image)
            else:
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
            if self._retry_with_cpu_fallback(exc):
                embedding = self.recognition_model.get_feat(face_image).flatten()
            else:
                wrapped = _wrap_model_runtime_error(exc, action="推理")
                logger.error("%s", wrapped)
                raise wrapped from exc
        return _l2_normalize(embedding)
