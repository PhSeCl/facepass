from src.eval.end2end_metrics import (
    GroundTruthFace,
    ImageMatchResult,
    PredictedFace,
    compute_metrics,
    iou,
    match_greedy,
)


def test_iou_handles_overlap_and_disjoint_boxes() -> None:
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == 50 / 150
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_iou_handles_containment_and_boundary_threshold() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 5, 5)) == 0.25
    assert iou((0, 0, 10, 10), (0, 0, 20, 10)) == 0.5


def test_match_greedy_handles_perfect_and_partial_matches() -> None:
    matches, unmatched_preds, unmatched_gts = match_greedy(
        pred_boxes=[(0, 0, 10, 10), (20, 0, 10, 10)],
        gt_boxes=[(0, 0, 10, 10), (40, 0, 10, 10)],
    )

    assert matches == [(0, 0)]
    assert unmatched_preds == [1]
    assert unmatched_gts == [1]


def test_match_greedy_resolves_competing_pairs_by_highest_iou() -> None:
    matches, unmatched_preds, unmatched_gts = match_greedy(
        pred_boxes=[(0, 0, 10, 10), (1, 0, 10, 10)],
        gt_boxes=[(0, 0, 10, 10)],
    )

    assert matches == [(0, 0)]
    assert unmatched_preds == [1]
    assert unmatched_gts == []


def test_compute_metrics_reports_detection_recognition_and_unknown_breakdowns() -> None:
    result = ImageMatchResult(
        predictions=[
            PredictedFace(bbox=(0, 0, 10, 10), identity_id="p01", is_unknown=False),
            PredictedFace(bbox=(20, 0, 10, 10), identity_id="p02", is_unknown=False),
            PredictedFace(bbox=(40, 0, 10, 10), identity_id="unknown", is_unknown=True),
            PredictedFace(bbox=(60, 0, 10, 10), identity_id="p01", is_unknown=False),
        ],
        ground_truths=[
            GroundTruthFace(bbox=(0, 0, 10, 10), identity_id="p01"),
            GroundTruthFace(bbox=(20, 0, 10, 10), identity_id="p03"),
            GroundTruthFace(bbox=(40, 0, 10, 10), identity_id="unknown"),
            GroundTruthFace(bbox=(80, 0, 10, 10), identity_id="p02"),
        ],
        matches=[(0, 0), (1, 1), (2, 2)],
        unmatched_prediction_indices=[3],
        unmatched_ground_truth_indices=[3],
    )

    metrics = compute_metrics([result])

    assert metrics.ground_truth_total == 4
    assert metrics.matched_ground_truth_total == 3
    assert metrics.false_positives == 1
    assert metrics.detection_recall == 0.75
    assert metrics.detection_precision == 0.75
    assert metrics.strict_top1_correct == 2
    assert metrics.strict_top1_accuracy == 0.5
    assert metrics.matched_top1_correct == 2
    assert metrics.matched_top1_accuracy == 2 / 3
    assert metrics.unknown_detected_total == 1
    assert metrics.unknown_detected_correct == 1
    assert metrics.unknown_detected_accuracy == 1.0
    assert metrics.predicted_unknown_total == 1
    assert metrics.predicted_unknown_correct == 1
    assert metrics.predicted_unknown_precision == 1.0
    assert metrics.confusion_pairs == [
        ("p01", "p01"),
        ("p03", "p02"),
        ("unknown", "unknown"),
    ]
