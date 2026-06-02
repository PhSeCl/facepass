from dataclasses import dataclass

from src.backend.gallery import Gallery
from src.backend.recognizer import Recognizer
from src.common.errors import InvalidImageError
from src.common.images import safe_load_image
from src.common.logging import get_logger
from src.face_model.base import FaceModel

from .end2end_dataset import GroupedSelfDataset
from .end2end_metrics import (
    EndToEndMetrics,
    ImageMatchResult,
    PredictedFace,
    compute_metrics,
    match_greedy,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class EndToEndEvalReport:
    image_results: list[ImageMatchResult]
    metrics: EndToEndMetrics


def evaluate_end2end(
    dataset: GroupedSelfDataset,
    model: FaceModel,
    threshold: float,
    id2name: dict[str, str] | None = None,
    gallery: Gallery | None = None,
) -> EndToEndEvalReport:
    active_gallery = gallery or Gallery()
    if gallery is None:
        active_gallery.build_from_dir(str(dataset.registered_root), model)
    recognizer = Recognizer(model, active_gallery, threshold, id2name or {})

    image_results: list[ImageMatchResult] = []
    for sample in dataset.images:
        try:
            image = safe_load_image(sample.image_path)
        except (InvalidImageError, OSError) as exc:
            logger.warning("跳过无效测试图 %s: %s", sample.image_path, exc)
            continue

        predictions = [
            PredictedFace(
                bbox=tuple(result.bbox),
                identity_id=result.identity_id,
                is_unknown=result.is_unknown,
            )
            for result in recognizer.recognize_image(image)
        ]
        matches, unmatched_prediction_indices, unmatched_ground_truth_indices = match_greedy(
            pred_boxes=[prediction.bbox for prediction in predictions],
            gt_boxes=[ground_truth.bbox for ground_truth in sample.faces],
        )

        for ground_truth_index in unmatched_ground_truth_indices:
            logger.warning(
                "missed detection: %s bbox=%s",
                sample.image_path.name,
                sample.faces[ground_truth_index].bbox,
            )
        for prediction_index in unmatched_prediction_indices:
            logger.warning(
                "false positive: %s bbox=%s",
                sample.image_path.name,
                predictions[prediction_index].bbox,
            )

        image_results.append(
            ImageMatchResult(
                predictions=predictions,
                ground_truths=sample.faces,
                matches=matches,
                unmatched_prediction_indices=unmatched_prediction_indices,
                unmatched_ground_truth_indices=unmatched_ground_truth_indices,
            )
        )

    return EndToEndEvalReport(
        image_results=image_results,
        metrics=compute_metrics(image_results),
    )
