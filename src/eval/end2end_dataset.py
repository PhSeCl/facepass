import json
import re
from dataclasses import dataclass
from pathlib import Path

from .end2end_metrics import GroundTruthFace


@dataclass(frozen=True)
class AnnotatedImage:
    image_path: Path
    faces: list[GroundTruthFace]


@dataclass(frozen=True)
class GroupedSelfDataset:
    registered_root: Path
    test_root: Path
    images: list[AnnotatedImage]


IDENTITY_PATTERN = re.compile(r"p\d{2}|unknown")


def _read_annotation_records(annotations_path: Path) -> list[tuple[str, object]]:
    suffix = annotations_path.suffix.lower()
    if suffix == ".jsonl":
        records: list[tuple[str, object]] = []
        for line_number, line in enumerate(annotations_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            records.append((f"第 {line_number} 行", json.loads(line)))
        return records
    if suffix == ".json":
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"标注文件格式无效: {annotations_path.name} 顶层必须是对象")
        return [(str(image_name), {"image": image_name, "faces": faces}) for image_name, faces in payload.items()]
    raise ValueError(f"不支持的标注文件格式: {annotations_path}")


def _resolve_image_path(test_root: Path, image_reference: str, source_label: str) -> Path:
    candidate = test_root / image_reference
    if candidate.exists():
        return candidate

    fallback = test_root / "images" / image_reference
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"标注引用的图片不存在: {image_reference} ({source_label})")


def _parse_bbox(value: object, source_label: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"无效 bbox: {source_label} 需要 [x, y, w, h]")
    return tuple(int(item) for item in value)


def _parse_identity(value: object, source_label: str) -> str:
    identity_id = str(value)
    if not IDENTITY_PATTERN.fullmatch(identity_id):
        raise ValueError(f"无效 identity: {identity_id} ({source_label})")
    return identity_id


def _parse_faces(raw_faces: object, source_label: str) -> list[GroundTruthFace]:
    if not isinstance(raw_faces, list):
        raise ValueError(f"无效 faces: {source_label} 必须是列表")

    faces: list[GroundTruthFace] = []
    for face_index, face in enumerate(raw_faces, start=1):
        if not isinstance(face, dict):
            raise ValueError(f"无效人脸标注: {source_label} 第 {face_index} 个 face 必须是对象")
        bbox = _parse_bbox(face.get("bbox"), f"{source_label} 第 {face_index} 个 face")
        identity_value = face["identity_id"] if "identity_id" in face else face.get("identity")
        if identity_value is None:
            raise ValueError(f"缺少 identity: {source_label} 第 {face_index} 个 face")
        faces.append(
            GroundTruthFace(
                bbox=bbox,
                identity_id=_parse_identity(identity_value, f"{source_label} 第 {face_index} 个 face"),
            )
        )
    return faces


def _parse_annotated_images(annotations_path: Path, test_root: Path) -> list[AnnotatedImage]:
    images: list[AnnotatedImage] = []
    for source_label, record in _read_annotation_records(annotations_path):
        if not isinstance(record, dict):
            raise ValueError(f"无效标注记录: {source_label} 必须是对象")
        image_reference = record["image_path"] if "image_path" in record else record.get("image")
        if image_reference is None:
            raise ValueError(f"缺少 image 字段: {source_label}")
        image_path = _resolve_image_path(test_root, str(image_reference), source_label)
        faces = _parse_faces(record.get("faces", []), str(image_path.name))
        images.append(
            AnnotatedImage(
                image_path=image_path,
                faces=faces,
            )
        )
    return images


def load_grouped_self_dataset(
    annotations_path: str | Path,
    test_root: str | Path,
    registered_root: str | Path | None = None,
) -> GroupedSelfDataset:
    annotations_path = Path(annotations_path)
    test_root = Path(test_root)
    registered_root = Path(registered_root) if registered_root is not None else test_root.parent / "registered"

    if not annotations_path.exists():
        raise FileNotFoundError(f"自采测试标注不存在: {annotations_path}")
    if not test_root.exists():
        raise FileNotFoundError(f"自采测试目录不存在: {test_root}")
    if not registered_root.exists():
        raise FileNotFoundError(f"自采注册集目录不存在: {registered_root}")

    return GroupedSelfDataset(
        registered_root=registered_root,
        test_root=test_root,
        images=_parse_annotated_images(annotations_path, test_root),
    )
