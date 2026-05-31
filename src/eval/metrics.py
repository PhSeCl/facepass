from dataclasses import dataclass


@dataclass(frozen=True)
class ClassAccuracy:
    correct: int
    total: int
    accuracy: float


@dataclass(frozen=True)
class AccuracyMetrics:
    total: int
    correct: int
    top1_accuracy: float
    per_class: dict[str, ClassAccuracy]


def compute_accuracy_metrics(pairs: list[tuple[str, str]]) -> AccuracyMetrics:
    total = len(pairs)
    correct = sum(1 for true_id, pred_id in pairs if true_id == pred_id)

    per_class_totals: dict[str, int] = {}
    per_class_correct: dict[str, int] = {}
    for true_id, pred_id in pairs:
        per_class_totals[true_id] = per_class_totals.get(true_id, 0) + 1
        if true_id == pred_id:
            per_class_correct[true_id] = per_class_correct.get(true_id, 0) + 1

    per_class: dict[str, ClassAccuracy] = {}
    for true_id, class_total in per_class_totals.items():
        class_correct = per_class_correct.get(true_id, 0)
        per_class[true_id] = ClassAccuracy(
            correct=class_correct,
            total=class_total,
            accuracy=class_correct / class_total if class_total else 0.0,
        )

    return AccuracyMetrics(
        total=total,
        correct=correct,
        top1_accuracy=correct / total if total else 0.0,
        per_class=per_class,
    )
