import pickle
from pathlib import Path

import numpy as np

from src.common.errors import EmptyGalleryError, InvalidImageError
from src.common.images import safe_load_image
from src.common.logging import get_logger
from src.face_model.base import FaceModel


logger = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("embedding must not be zero")
    return vector / norm


class Gallery:
    def __init__(self, entries: dict[str, list[np.ndarray]] | None = None) -> None:
        self._entries: dict[str, list[np.ndarray]] = entries or {}

    def register(self, identity_id: str, embeddings: list[np.ndarray]) -> None:
        if not embeddings:
            return
        self._entries.setdefault(identity_id, [])
        self._entries[identity_id].extend(_normalize(embedding) for embedding in embeddings)

    def build_from_dir(self, root: str, model: FaceModel) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise EmptyGalleryError(f"注册集目录不存在: {root_path}")

        empty_identities: list[str] = []
        for identity_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
            before = len(self._entries.get(identity_dir.name, []))
            for image_path in sorted(identity_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    image = safe_load_image(image_path)
                    embedding = model.encode_aligned(image)
                    self.register(identity_dir.name, [embedding])
                except (InvalidImageError, OSError, ValueError) as exc:
                    logger.warning("跳过无效注册图 %s: %s", image_path, exc)
            after = len(self._entries.get(identity_dir.name, []))
            if after == before:
                empty_identities.append(identity_dir.name)

        if empty_identities:
            logger.warning("以下身份没有有效注册图: %s", ", ".join(empty_identities))
        if not self._entries:
            raise EmptyGalleryError("身份库为空，至少需要一张有效注册图")

    def match(self, embedding: np.ndarray) -> tuple[str, float] | None:
        if not self._entries:
            return None
        query = _normalize(embedding)
        best_identity: str | None = None
        best_similarity = -1.0
        for identity_id, embeddings in self._entries.items():
            if not embeddings:
                continue
            similarities = [float(np.dot(query, candidate)) for candidate in embeddings]
            identity_similarity = max(similarities)
            if identity_similarity > best_similarity:
                best_similarity = identity_similarity
                best_identity = identity_id
        if best_identity is None:
            return None
        return best_identity, max(0.0, best_similarity)

    def identities(self) -> list[dict[str, int | str]]:
        return [
            {"identity_id": identity_id, "count": len(embeddings)}
            for identity_id, embeddings in sorted(self._entries.items())
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self._entries, handle)

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        path = Path(path)
        with path.open("rb") as handle:
            entries = pickle.load(handle)
        return cls(entries=entries)
