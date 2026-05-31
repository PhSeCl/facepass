import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.errors import InvalidImageError
from src.common.images import safe_load_image
from src.common.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class SelfSample:
    face_image: np.ndarray
    identity_id: str
    source_path: Path
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class SelfDataset:
    registered_root: Path
    test_root: Path
    samples: list[SelfSample]


def _crop_face(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    height, width = image.shape[:2]
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(x1, min(width, x + max(0, w)))
    y2 = max(y1, min(height, y + max(0, h)))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise InvalidImageError(f"人脸框无效: {bbox}")
    return crop


def load_self_dataset(
    annotations_path: str | Path,
    test_root: str | Path,
    registered_root: str | Path | None = None,
) -> SelfDataset:
    annotations_path = Path(annotations_path)
    test_root = Path(test_root)
    registered_root = Path(registered_root) if registered_root is not None else test_root.parent / "registered"

    if not annotations_path.exists():
        raise FileNotFoundError(f"自采测试标注不存在: {annotations_path}")
    if not test_root.exists():
        raise FileNotFoundError(f"自采测试目录不存在: {test_root}")
    if not registered_root.exists():
        raise FileNotFoundError(f"自采注册集目录不存在: {registered_root}")

    samples: list[SelfSample] = []
    for line_number, line in enumerate(annotations_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        image_path = test_root / record["image_path"]
        try:
            image = safe_load_image(image_path)
        except (InvalidImageError, OSError) as exc:
            logger.warning("跳过无效测试图 %s: %s", image_path, exc)
            continue

        for face in record.get("faces", []):
            bbox = tuple(int(value) for value in face["bbox"])
            try:
                face_image = _crop_face(image, bbox)
            except InvalidImageError as exc:
                logger.warning("跳过无效人脸框 %s 第 %s 行: %s", image_path, line_number, exc)
                continue
            samples.append(
                SelfSample(
                    face_image=face_image,
                    identity_id=str(face["identity_id"]),
                    source_path=image_path,
                    bbox=bbox,
                )
            )

    return SelfDataset(
        registered_root=registered_root,
        test_root=test_root,
        samples=samples,
    )
