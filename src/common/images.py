from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

from .errors import InvalidImageError


MAX_IMAGE_PIXELS = 25_000_000


def _read_bytes(path_or_bytes: str | Path | bytes | bytearray | BinaryIO) -> bytes:
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise InvalidImageError(f"无法读取图片文件: {path}") from exc
    elif isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    elif hasattr(path_or_bytes, "read"):
        data = path_or_bytes.read()
    else:
        raise InvalidImageError("不支持的图片输入类型")

    if not data:
        raise InvalidImageError("图片为空或 0 字节")
    return data


def safe_load_image(path_or_bytes: str | Path | bytes | bytearray | BinaryIO) -> np.ndarray:
    """Decode image bytes/path into a BGR numpy array.

    Invalid image formats, empty files, unsupported inputs, and extremely large
    images raise InvalidImageError. These are deterministic recoverable errors
    and should not be retried.
    """
    data = _read_bytes(path_or_bytes)
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidImageError("图片格式损坏或不受支持")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise InvalidImageError("图片尺寸无效")
    if height * width > MAX_IMAGE_PIXELS:
        raise InvalidImageError(f"图片尺寸过大，最大允许 {MAX_IMAGE_PIXELS} 像素")
    return image
