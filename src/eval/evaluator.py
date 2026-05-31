from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from src.backend.gallery import Gallery

from .metrics import AccuracyMetrics, compute_accuracy_metrics


@dataclass(frozen=True)
class EvalSample:
    true_identity_id: str
    predicted_identity_id: str
    similarity: float


@dataclass(frozen=True)
class EvalReport:
    metrics: AccuracyMetrics
    samples: list[EvalSample]


def evaluate(
    model,
    gallery: Gallery,
    samples: Iterable[tuple[np.ndarray, str]],
    threshold: float | None = None,
) -> EvalReport:
    results: list[EvalSample] = []
    for face_image, true_identity_id in samples:
        embedding = model.encode_aligned(face_image)
        match = gallery.match(embedding)
        if match is None:
            predicted_identity_id = "unknown"
            similarity = 0.0
        else:
            best_identity_id, similarity = match
            # Keep the evaluation layer independent from Recognizer while
            # mirroring its unknown-threshold behavior for comparable reports.
            predicted_identity_id = (
                best_identity_id
                if threshold is None or similarity >= threshold
                else "unknown"
            )
        results.append(
            EvalSample(
                true_identity_id=true_identity_id,
                predicted_identity_id=predicted_identity_id,
                similarity=float(similarity),
            )
        )

    metrics = compute_accuracy_metrics(
        [(item.true_identity_id, item.predicted_identity_id) for item in results]
    )
    return EvalReport(metrics=metrics, samples=results)
