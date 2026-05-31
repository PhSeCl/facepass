from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from src.common.errors import InvalidImageError
from src.common.images import safe_load_image
from src.common.logging import get_logger


logger = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("embedding must not be zero")
    return vector / norm


@dataclass(frozen=True)
class SimilarityDistributions:
    same_identity: list[float]
    different_identity: list[float]


def collect_similarity_distributions(
    embeddings_by_identity: dict[str, list[np.ndarray]],
) -> SimilarityDistributions:
    normalized = {
        identity_id: [_normalize(embedding) for embedding in embeddings]
        for identity_id, embeddings in embeddings_by_identity.items()
        if embeddings
    }

    same_identity: list[float] = []
    for embeddings in normalized.values():
        for left, right in combinations(embeddings, 2):
            same_identity.append(float(np.dot(left, right)))

    different_identity: list[float] = []
    identity_ids = sorted(normalized)
    for left_index, left_identity in enumerate(identity_ids):
        for right_identity in identity_ids[left_index + 1 :]:
            for left_embedding in normalized[left_identity]:
                for right_embedding in normalized[right_identity]:
                    different_identity.append(float(np.dot(left_embedding, right_embedding)))

    return SimilarityDistributions(
        same_identity=same_identity,
        different_identity=different_identity,
    )


def suggest_threshold(
    distributions: SimilarityDistributions,
    target_false_accept_rate: float | None = None,
) -> float:
    same_identity = distributions.same_identity
    different_identity = distributions.different_identity
    if not same_identity or not different_identity:
        raise ValueError("same_identity 和 different_identity 分布都必须非空")

    if target_false_accept_rate is not None:
        if not 0.0 <= target_false_accept_rate <= 1.0:
            raise ValueError("target_false_accept_rate 必须在 [0, 1] 范围内")
        candidates = sorted(set(different_identity + same_identity))
        chosen = candidates[-1]
        for candidate in candidates:
            false_accept_rate = sum(score >= candidate for score in different_identity) / len(different_identity)
            if false_accept_rate <= target_false_accept_rate:
                chosen = candidate
                break
        return float(chosen)

    values = sorted(set(different_identity + same_identity))
    candidate_thresholds = [values[0] - 1e-6]
    candidate_thresholds.extend(
        (left + right) / 2.0 for left, right in zip(values, values[1:], strict=False)
    )
    candidate_thresholds.append(values[-1] + 1e-6)

    best_threshold = candidate_thresholds[0]
    best_errors = None
    for threshold in candidate_thresholds:
        false_rejects = sum(score < threshold for score in same_identity)
        false_accepts = sum(score >= threshold for score in different_identity)
        errors = false_rejects + false_accepts
        if best_errors is None or errors < best_errors:
            best_errors = errors
            best_threshold = threshold
    return float(best_threshold)


def collect_registered_embeddings(registered_root: str | Path, model) -> dict[str, list[np.ndarray]]:
    registered_root = Path(registered_root)
    if not registered_root.exists():
        raise FileNotFoundError(f"注册集目录不存在: {registered_root}")

    embeddings_by_identity: dict[str, list[np.ndarray]] = {}
    for identity_dir in sorted(path for path in registered_root.iterdir() if path.is_dir()):
        embeddings: list[np.ndarray] = []
        for image_path in sorted(identity_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                image = safe_load_image(image_path)
                embeddings.append(model.encode_aligned(image))
            except (InvalidImageError, OSError, ValueError) as exc:
                logger.warning("跳过无效注册图 %s: %s", image_path, exc)
        if embeddings:
            embeddings_by_identity[identity_dir.name] = embeddings

    if not embeddings_by_identity:
        raise ValueError("注册集中没有可用 embedding")
    return embeddings_by_identity
