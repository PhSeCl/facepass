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


def _deduplicate_registered_identities(
    faces: list[dict[str, object]],
    review_reasons: list[str],
) -> None:
    best_index_by_identity: dict[str, int] = {}
    for index, face in enumerate(faces):
        identity = face["identity"]
        if not isinstance(identity, str) or identity == "unknown":
            continue
        score = float(face["score"])
        current_best_index = best_index_by_identity.get(identity)
        if current_best_index is None:
            best_index_by_identity[identity] = index
            continue
        current_best_score = float(faces[current_best_index]["score"])
        if score > current_best_score:
            best_index_by_identity[identity] = index

    duplicate_identities = [
        identity
        for identity in best_index_by_identity
        if sum(1 for face in faces if face["identity"] == identity) > 1
    ]
    for identity in duplicate_identities:
        best_index = best_index_by_identity[identity]
        kept_face = faces[best_index]
        for index, face in enumerate(faces):
            if index == best_index or face["identity"] != identity:
                continue
            face["identity"] = "unknown"
        review_reasons.append(
            "同图存在相似人物，"
            f"检测到多个 face 命中 {identity}；"
            f"预标注暂保留最高分 bbox={kept_face['bbox']} score={float(kept_face['score']):.4f}，"
            "其余同身份候选已改为 unknown，请人工核对"
        )


def generate_draft_annotations(
    recognizer,
    images_dir: str | Path,
    out_path: str | Path,
    review_threshold: float,
    draft_identity_threshold: float,
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
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    processed_images = 0
    total_faces = 0
    review_images = 0

    for image_path in image_paths:
        if image_path.name in payload:
            print(f"{image_path.name}: skipped=existing-annotation")
            continue

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
            draft_identity = (
                preview.best_identity_id
                if preview.best_identity_id and preview.similarity >= draft_identity_threshold
                else "unknown"
            )
            if preview.best_identity_id:
                faces.append(
                    {
                        "bbox": list(preview.bbox),
                        "identity": draft_identity,
                        "score": round(float(preview.similarity), 6),
                    }
                )
            if preview.similarity < review_threshold:
                needs_review = True
                review_reasons.append(
                    f"低置信度 bbox={list(preview.bbox)} score={preview.similarity:.4f}"
                )

        _deduplicate_registered_identities(faces, review_reasons)
        if any("同图存在相似人物" in reason for reason in review_reasons):
            needs_review = True

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
