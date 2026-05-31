from dataclasses import dataclass


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class PredictedFace:
    bbox: BBox
    identity_id: str
    is_unknown: bool


@dataclass(frozen=True)
class GroundTruthFace:
    bbox: BBox
    identity_id: str


@dataclass(frozen=True)
class ImageMatchResult:
    predictions: list[PredictedFace]
    ground_truths: list[GroundTruthFace]
    matches: list[tuple[int, int]]
    unmatched_prediction_indices: list[int]
    unmatched_ground_truth_indices: list[int]


@dataclass(frozen=True)
class EndToEndMetrics:
    ground_truth_total: int
    matched_ground_truth_total: int
    false_positives: int
    detection_recall: float
    detection_precision: float
    strict_top1_correct: int
    strict_top1_accuracy: float
    matched_top1_correct: int
    matched_top1_accuracy: float
    unknown_detected_total: int
    unknown_detected_correct: int
    unknown_detected_accuracy: float
    predicted_unknown_total: int
    predicted_unknown_correct: int
    predicted_unknown_precision: float
    confusion_pairs: list[tuple[str, str]]


def _to_corners(box: BBox) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + max(0, int(w))
    y2 = y1 + max(0, int(h))
    return x1, y1, x2, y2


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = _to_corners(box_a)
    bx1, by1, bx2, by2 = _to_corners(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def match_greedy(
    pred_boxes: list[BBox],
    gt_boxes: list[BBox],
    iou_thr: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            overlap = iou(pred_box, gt_box)
            if overlap >= iou_thr:
                candidates.append((overlap, pred_idx, gt_idx))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _, pred_idx, gt_idx in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx))

    unmatched_predictions = [index for index in range(len(pred_boxes)) if index not in used_pred]
    unmatched_ground_truths = [index for index in range(len(gt_boxes)) if index not in used_gt]
    return matches, unmatched_predictions, unmatched_ground_truths


def compute_metrics(results: list[ImageMatchResult]) -> EndToEndMetrics:
    ground_truth_total = sum(len(item.ground_truths) for item in results)
    matched_ground_truth_total = sum(len(item.matches) for item in results)
    false_positives = sum(len(item.unmatched_prediction_indices) for item in results)
    prediction_total = sum(len(item.predictions) for item in results)

    strict_top1_correct = 0
    matched_top1_correct = 0
    unknown_detected_total = 0
    unknown_detected_correct = 0
    predicted_unknown_total = 0
    predicted_unknown_correct = 0
    confusion_pairs: list[tuple[str, str]] = []

    for result in results:
        for pred_idx, gt_idx in result.matches:
            prediction = result.predictions[pred_idx]
            ground_truth = result.ground_truths[gt_idx]
            confusion_pairs.append((ground_truth.identity_id, prediction.identity_id))

            is_correct = prediction.identity_id == ground_truth.identity_id
            if is_correct:
                strict_top1_correct += 1
                matched_top1_correct += 1
            if ground_truth.identity_id == "unknown":
                unknown_detected_total += 1
                if prediction.is_unknown:
                    unknown_detected_correct += 1
            if prediction.is_unknown:
                predicted_unknown_total += 1
                if ground_truth.identity_id == "unknown":
                    predicted_unknown_correct += 1

    detection_recall = (
        matched_ground_truth_total / ground_truth_total if ground_truth_total else 0.0
    )
    detection_precision = matched_ground_truth_total / prediction_total if prediction_total else 0.0
    strict_top1_accuracy = strict_top1_correct / ground_truth_total if ground_truth_total else 0.0
    matched_top1_accuracy = (
        matched_top1_correct / matched_ground_truth_total if matched_ground_truth_total else 0.0
    )
    unknown_detected_accuracy = (
        unknown_detected_correct / unknown_detected_total if unknown_detected_total else 0.0
    )
    predicted_unknown_precision = (
        predicted_unknown_correct / predicted_unknown_total if predicted_unknown_total else 0.0
    )

    return EndToEndMetrics(
        ground_truth_total=ground_truth_total,
        matched_ground_truth_total=matched_ground_truth_total,
        false_positives=false_positives,
        detection_recall=detection_recall,
        detection_precision=detection_precision,
        strict_top1_correct=strict_top1_correct,
        strict_top1_accuracy=strict_top1_accuracy,
        matched_top1_correct=matched_top1_correct,
        matched_top1_accuracy=matched_top1_accuracy,
        unknown_detected_total=unknown_detected_total,
        unknown_detected_correct=unknown_detected_correct,
        unknown_detected_accuracy=unknown_detected_accuracy,
        predicted_unknown_total=predicted_unknown_total,
        predicted_unknown_correct=predicted_unknown_correct,
        predicted_unknown_precision=predicted_unknown_precision,
        confusion_pairs=confusion_pairs,
    )
