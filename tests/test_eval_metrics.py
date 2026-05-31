from src.eval.metrics import ClassAccuracy, compute_accuracy_metrics


def test_compute_accuracy_metrics_reports_perfect_accuracy() -> None:
    metrics = compute_accuracy_metrics(
        [
            ("p01", "p01"),
            ("p02", "p02"),
        ]
    )

    assert metrics.total == 2
    assert metrics.correct == 2
    assert metrics.top1_accuracy == 1.0
    assert metrics.per_class == {
        "p01": ClassAccuracy(correct=1, total=1, accuracy=1.0),
        "p02": ClassAccuracy(correct=1, total=1, accuracy=1.0),
    }


def test_compute_accuracy_metrics_reports_zero_accuracy() -> None:
    metrics = compute_accuracy_metrics(
        [
            ("p01", "p02"),
            ("p02", "p01"),
        ]
    )

    assert metrics.total == 2
    assert metrics.correct == 0
    assert metrics.top1_accuracy == 0.0
    assert metrics.per_class["p01"] == ClassAccuracy(correct=0, total=1, accuracy=0.0)
    assert metrics.per_class["p02"] == ClassAccuracy(correct=0, total=1, accuracy=0.0)


def test_compute_accuracy_metrics_reports_mixed_per_class_breakdown() -> None:
    metrics = compute_accuracy_metrics(
        [
            ("p01", "p01"),
            ("p01", "p02"),
            ("p02", "p02"),
            ("p03", "unknown"),
        ]
    )

    assert metrics.total == 4
    assert metrics.correct == 2
    assert metrics.top1_accuracy == 0.5
    assert metrics.per_class["p01"] == ClassAccuracy(correct=1, total=2, accuracy=0.5)
    assert metrics.per_class["p02"] == ClassAccuracy(correct=1, total=1, accuracy=1.0)
    assert metrics.per_class["p03"] == ClassAccuracy(correct=0, total=1, accuracy=0.0)


def test_compute_accuracy_metrics_handles_empty_input() -> None:
    metrics = compute_accuracy_metrics([])

    assert metrics.total == 0
    assert metrics.correct == 0
    assert metrics.top1_accuracy == 0.0
    assert metrics.per_class == {}
