import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.common.errors import InvalidImageError
from src.common.images import safe_load_image
from src.common.logging import get_logger


logger = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SINGLE_PERSON_PATTERN = re.compile(r"p\d{2}_t\d+\Z", re.IGNORECASE)


@dataclass(frozen=True)
class PreannotationSummary:
    processed_images: int
    total_faces: int
    review_images: int


def _iter_image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _needs_single_person_review(path: Path, face_count: int) -> bool:
    return face_count > 1 and SINGLE_PERSON_PATTERN.fullmatch(path.stem) is not None


def generate_draft_annotations(
    recognizer,
    images_dir: str | Path,
    out_path: str | Path,
    review_threshold: float,
    overwrite: bool = False,
) -> PreannotationSummary:
    images_dir = Path(images_dir)
    out_path = Path(out_path)

    if not images_dir.exists():
        raise FileNotFoundError(f"测试图片目录不存在: {images_dir}")
    if not images_dir.is_dir():
        raise ValueError(f"测试图片路径不是目录: {images_dir}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"标注文件已存在: {out_path}")

    image_paths = _iter_image_paths(images_dir)
    if not image_paths:
        raise ValueError(f"测试图片目录为空: {images_dir}")

    payload: dict[str, list[dict[str, object]]] = {}
    processed_images = 0
    total_faces = 0
    review_images = 0

    for image_path in image_paths:
        try:
            image = safe_load_image(image_path)
        except (InvalidImageError, OSError) as exc:
            logger.warning("跳过无效测试图 %s: %s", image_path, exc)
            continue

        previews = recognizer.preview_image(image)
        faces: list[dict[str, object]] = []
        needs_review = False
        review_reasons: list[str] = []
        for preview in previews:
            if preview.best_identity_id:
                faces.append(
                    {
                        "bbox": list(preview.bbox),
                        "identity": preview.best_identity_id,
                        "score": round(float(preview.similarity), 6),
                    }
                )
            if preview.similarity < review_threshold:
                needs_review = True
                review_reasons.append(
                    f"低置信度 bbox={list(preview.bbox)} score={preview.similarity:.4f}"
                )

        face_count = len(previews)
        if face_count == 0:
            needs_review = True
            review_reasons.append("未检出人脸，核对图片质量")
        if _needs_single_person_review(image_path, face_count):
            needs_review = True
            review_reasons.append("单人照检出多脸，核对是否有路人或误检")
        if face_count > 6:
            needs_review = True
            review_reasons.append("检出人数较多，核对是否超出合照规范或有误检")

        payload[image_path.name] = faces
        processed_images += 1
        total_faces += face_count
        if needs_review:
            review_images += 1
            for reason in review_reasons:
                logger.warning("%s: %s", image_path.name, reason)

        print(
            f"{image_path.name}: faces={face_count}, "
            f"review={'yes' if needs_review else 'no'}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return PreannotationSummary(
        processed_images=processed_images,
        total_faces=total_faces,
        review_images=review_images,
    )
