import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from src.common.errors import EmptyGalleryError, InvalidImageError
from src.common.images import safe_load_image
from src.common.logging import get_logger
from src.face_model.base import FaceModel


logger = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GALLERY_FORMAT_VERSION = 3


def _normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("embedding must not be zero")
    return vector / norm


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    _, _, width, height = bbox
    return max(0, int(width)) * max(0, int(height))


class Gallery:
    def __init__(
        self,
        entries: dict[str, list[np.ndarray]] | None = None,
        *,
        metadata: dict[str, dict[str, int]] | None = None,
        requires_rebuild: bool = False,
    ) -> None:
        self._entries: dict[str, list[np.ndarray]] = entries or {}
        self._metadata: dict[str, dict[str, int]] = metadata or {}
        self.requires_rebuild = requires_rebuild
        self._matrix: np.ndarray | None = None
        self._index: list[tuple[str, int]] = []
        if self._entries:
            self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        rows: list[np.ndarray] = []
        index: list[tuple[str, int]] = []
        for identity_id, embeddings in self._entries.items():
            for i, emb in enumerate(embeddings):
                rows.append(emb)
                index.append((identity_id, i))
        if rows:
            self._matrix = np.stack(rows, axis=0)
            self._index = index
        else:
            self._matrix = None
            self._index = []

    def register(
        self,
        identity_id: str,
        embeddings: list[np.ndarray],
        *,
        valid_image_count: int | None = None,
    ) -> None:
        if not embeddings:
            return
        normalized_embeddings = [_normalize(embedding) for embedding in embeddings]
        self._entries.setdefault(identity_id, [])
        self._entries[identity_id].extend(normalized_embeddings)
        stats = self._metadata.setdefault(
            identity_id,
            {"prototype_count": 0, "valid_image_count": 0},
        )
        stats["prototype_count"] = len(self._entries[identity_id])
        stats["valid_image_count"] += valid_image_count if valid_image_count is not None else len(normalized_embeddings)
        self._rebuild_matrix()

    def build_from_dir(self, root: str, model: FaceModel) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise EmptyGalleryError(f"注册集目录不存在: {root_path}")

        empty_identities: list[str] = []
        for identity_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
            identity_embeddings: list[np.ndarray] = []
            for image_path in sorted(identity_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    image = safe_load_image(image_path)
                    detected_faces = model.detect_and_encode(image)
                    if not detected_faces:
                        logger.warning("跳过未检测到人脸的注册图 %s", image_path)
                        continue
                    if len(detected_faces) > 1:
                        logger.warning("注册图 %s 检测到多张人脸，将使用面积最大的一张", image_path)
                    selected_face = max(detected_faces, key=lambda face: _bbox_area(face.bbox))
                    identity_embeddings.append(selected_face.embedding)
                except (InvalidImageError, OSError, ValueError) as exc:
                    logger.warning("跳过无效注册图 %s: %s", image_path, exc)
            if not identity_embeddings:
                empty_identities.append(identity_dir.name)
                continue

            averaged_embedding = np.mean(np.stack(identity_embeddings, axis=0), axis=0)
            self.register(
                identity_dir.name,
                [averaged_embedding],
                valid_image_count=len(identity_embeddings),
            )

        if empty_identities:
            logger.warning("以下身份没有有效注册图: %s", ", ".join(empty_identities))
        if not self._entries:
            raise EmptyGalleryError("身份库为空，至少需要一张有效注册图")

    def build_from_cropped_dir(self, root: str, model: FaceModel) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise EmptyGalleryError(f"注册集目录不存在: {root_path}")

        empty_identities: list[str] = []
        for identity_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
            identity_embeddings: list[np.ndarray] = []
            for image_path in sorted(identity_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    image = safe_load_image(image_path)
                    identity_embeddings.append(model.encode_aligned(image))
                except (InvalidImageError, OSError, ValueError) as exc:
                    logger.warning("跳过无效注册图 %s: %s", image_path, exc)
            if not identity_embeddings:
                empty_identities.append(identity_dir.name)
                continue

            averaged_embedding = np.mean(np.stack(identity_embeddings, axis=0), axis=0)
            self.register(
                identity_dir.name,
                [averaged_embedding],
                valid_image_count=len(identity_embeddings),
            )

        if empty_identities:
            logger.warning("以下身份没有有效注册图: %s", ", ".join(empty_identities))
        if not self._entries:
            raise EmptyGalleryError("身份库为空，至少需要一张有效注册图")

    def match(self, embedding: np.ndarray) -> tuple[str, float] | None:
        if not self._entries or self._matrix is None:
            return None
        query = _normalize(embedding)
        similarities = query @ self._matrix.T
        best_per_identity: dict[str, float] = {}
        for idx, (identity_id, _) in enumerate(self._index):
            sim = float(similarities[idx])
            if identity_id not in best_per_identity or sim > best_per_identity[identity_id]:
                best_per_identity[identity_id] = sim
        if not best_per_identity:
            return None
        best_identity = max(best_per_identity, key=best_per_identity.get)  # type: ignore[arg-type]
        return best_identity, max(0.0, best_per_identity[best_identity])

    def identities(self) -> list[dict[str, int | str]]:
        rows: list[dict[str, int | str]] = []
        for identity_id, embeddings in sorted(self._entries.items()):
            stats = self._metadata.get(identity_id, {})
            prototype_count = len(embeddings)
            rows.append(
                {
                    "identity_id": identity_id,
                    "count": prototype_count,
                    "prototype_count": prototype_count,
                    "valid_image_count": int(stats.get("valid_image_count", prototype_count)),
                }
            )
        return rows

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten entries into arrays for npz serialization
        identity_ids: list[str] = []
        offsets: list[int] = []
        all_embeddings: list[np.ndarray] = []
        offset = 0
        for identity_id in sorted(self._entries):
            embs = self._entries[identity_id]
            identity_ids.append(identity_id)
            offsets.append(offset)
            all_embeddings.extend(embs)
            offset += len(embs)

        embeddings_matrix = np.stack(all_embeddings, axis=0) if all_embeddings else np.empty((0, 0), dtype=np.float32)
        metadata_json = json.dumps(
            {"version": GALLERY_FORMAT_VERSION, "metadata": self._metadata},
            ensure_ascii=False,
        )

        buf = io.BytesIO()
        np.savez(
            buf,
            identity_ids=np.array(identity_ids, dtype=str),
            embeddings=embeddings_matrix,
            offsets=np.array(offsets, dtype=np.int64),
            metadata_json=np.array(metadata_json),
        )
        npz_bytes = buf.getvalue()

        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        tmp_path: str | None = tmp_name
        try:
            # A buffered file object's write() drains the whole buffer (looping
            # internally), unlike a single os.write() which may write fewer
            # bytes and leave a truncated cache behind once os.replace() runs.
            with os.fdopen(tmp_fd, "wb") as handle:
                handle.write(npz_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, str(path))
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        path = Path(path)

        # Read everything inside the context manager so the underlying file
        # handle is always closed before we return. Leaving an NpzFile open
        # would let a later atomic save (os.replace onto the same path) fail
        # with PermissionError on Windows.
        try:
            with np.load(str(path), allow_pickle=False) as data:
                identity_ids = list(data["identity_ids"])
                embeddings = np.asarray(data["embeddings"])
                offsets = list(data["offsets"])
                meta_obj = json.loads(str(data["metadata_json"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            # Old pickle caches, unreadable, or malformed files cannot be
            # trusted; signal rebuild instead of unpickling arbitrary payloads.
            logger.warning("无法加载 gallery 缓存，需要重建: %s (%s)", path, exc)
            return cls(requires_rebuild=True)

        if meta_obj.get("version") != GALLERY_FORMAT_VERSION:
            logger.warning("Gallery 缓存版本不匹配，需要重建: %s", path)
            return cls(requires_rebuild=True)

        metadata = meta_obj.get("metadata", {})

        # Reconstruct entries dict
        entries: dict[str, list[np.ndarray]] = {}
        for i, identity_id in enumerate(identity_ids):
            start = offsets[i]
            end = offsets[i + 1] if i + 1 < len(offsets) else embeddings.shape[0]
            entries[identity_id] = [embeddings[j] for j in range(start, end)]

        return cls(entries=entries, metadata=metadata)
