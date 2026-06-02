from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import AccuracyMetrics


def plot_top1_accuracy(metrics: AccuracyMetrics, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    bar = plt.bar(["top-1"], [metrics.top1_accuracy], color=["#457b9d"])[0]
    plt.ylim(0.0, 1.05)
    plt.ylabel("accuracy")
    plt.title("CelebA Top-1 Accuracy")
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        metrics.top1_accuracy + 0.02,
        f"{metrics.top1_accuracy:.3f}",
        ha="center",
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_per_class_accuracy(metrics: AccuracyMetrics, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(metrics.per_class.keys())
    values = [class_metrics.accuracy for class_metrics in metrics.per_class.values()]

    width = max(8, len(labels) * 0.4)
    plt.figure(figsize=(width, 4.5))
    bars = plt.bar(labels, values, color="#2a9d8f")
    plt.ylim(0.0, 1.05)
    plt.ylabel("accuracy")
    plt.title("CelebA Per-Class Accuracy")
    plt.xticks(rotation=45, ha="right")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
