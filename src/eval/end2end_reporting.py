from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .end2end_dataset import GroupedSelfDataset
from .end2end_evaluator import EndToEndEvalReport


def write_report(report_path: str | Path, report: EndToEndEvalReport) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = report.metrics
    payload = {
        "ground_truth_total": metrics.ground_truth_total,
        "matched_ground_truth_total": metrics.matched_ground_truth_total,
        "false_positives": metrics.false_positives,
        "detection_recall": metrics.detection_recall,
        "detection_precision": metrics.detection_precision,
        "strict_top1_correct": metrics.strict_top1_correct,
        "strict_top1_accuracy": metrics.strict_top1_accuracy,
        "matched_top1_correct": metrics.matched_top1_correct,
        "matched_top1_accuracy": metrics.matched_top1_accuracy,
        "unknown_detected_total": metrics.unknown_detected_total,
        "unknown_detected_correct": metrics.unknown_detected_correct,
        "unknown_detected_accuracy": metrics.unknown_detected_accuracy,
        "predicted_unknown_total": metrics.predicted_unknown_total,
        "predicted_unknown_correct": metrics.predicted_unknown_correct,
        "predicted_unknown_precision": metrics.predicted_unknown_precision,
        "confusion_pairs": [
            {
                "true_identity_id": true_identity_id,
                "predicted_identity_id": predicted_identity_id,
            }
            for true_identity_id, predicted_identity_id in metrics.confusion_pairs
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _identity_labels(dataset: GroupedSelfDataset, report: EndToEndEvalReport) -> list[str]:
    labels = {"unknown"}
    for sample in dataset.images:
        labels.update(face.identity_id for face in sample.faces)
    for true_identity_id, predicted_identity_id in report.metrics.confusion_pairs:
        labels.add(true_identity_id)
        labels.add(predicted_identity_id)
    return sorted(labels)


def plot_confusion_matrix(
    dataset: GroupedSelfDataset,
    report: EndToEndEvalReport,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = _identity_labels(dataset, report)
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for true_identity_id, predicted_identity_id in report.metrics.confusion_pairs:
        matrix[label_to_index[true_identity_id], label_to_index[predicted_identity_id]] += 1

    plt.figure(figsize=(max(5, len(labels) * 1.2), max(4, len(labels) * 1.0)))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="count")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("predicted identity")
    plt.ylabel("true identity")
    plt.title("End-to-End Confusion Matrix")
    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            plt.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_detection_metrics(report: EndToEndEvalReport, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["recall", "precision"]
    values = [
        report.metrics.detection_recall,
        report.metrics.detection_precision,
    ]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#2a9d8f", "#457b9d"])
    plt.ylim(0.0, 1.05)
    plt.ylabel("score")
    plt.title("Detection Metrics")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_metrics(report: EndToEndEvalReport, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["strict", "lenient"]
    values = [
        report.metrics.strict_top1_accuracy,
        report.metrics.matched_top1_accuracy,
    ]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#e76f51", "#f4a261"])
    plt.ylim(0.0, 1.05)
    plt.ylabel("accuracy")
    plt.title("End-to-End Accuracy")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
