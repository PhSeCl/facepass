import json
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

    images: list[AnnotatedImage] = []
    for line in annotations_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        faces = [
            GroundTruthFace(
                bbox=tuple(int(value) for value in face["bbox"]),
                identity_id=str(face["identity_id"]),
            )
            for face in record.get("faces", [])
        ]
        images.append(
            AnnotatedImage(
                image_path=test_root / record["image_path"],
                faces=faces,
            )
        )

    return GroupedSelfDataset(
        registered_root=registered_root,
        test_root=test_root,
        images=images,
    )
