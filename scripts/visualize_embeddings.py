import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from src.common.errors import InvalidImageError
from src.common.images import safe_load_image
from src.face_model import create_model


@dataclass(frozen=True)
class EmbeddingPoint:
    embedding: np.ndarray
    true_identity_id: str
    predicted_identity_id: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize report embeddings with t-SNE.")
    parser.add_argument(
        "--report-path",
        required=True,
        help="Path to an evaluation JSON report containing samples[].",
    )
    parser.add_argument(
        "--output-path",
        default="reports/tsne.png",
        help="Path for the output PNG figure.",
    )
    parser.add_argument(
        "--model-name",
        default="insightface",
        help="Model factory name passed to create_model().",
    )
    return parser


def _crop_if_needed(image: np.ndarray, bbox: list[int] | tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return image
    x, y, w, h = [int(value) for value in bbox]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = max(x1, x + max(0, w))
    y2 = max(y1, y + max(0, h))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise InvalidImageError(f"人脸框无效: {bbox}")
    return crop


def load_points(report_path: str | Path, model) -> list[EmbeddingPoint]:
    report_path = Path(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    points: list[EmbeddingPoint] = []
    for item in payload.get("samples", []):
        image = safe_load_image(item["path"])
        face_image = _crop_if_needed(image, item.get("bbox"))
        points.append(
            EmbeddingPoint(
                embedding=model.encode_aligned(face_image),
                true_identity_id=str(item["true_identity_id"]),
                predicted_identity_id=str(item["predicted_identity_id"]),
            )
        )
    if len(points) < 2:
        raise ValueError("至少需要两个样本才能做 t-SNE 可视化")
    return points


def write_tsne(points: list[EmbeddingPoint], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.stack([point.embedding for point in points], axis=0)
    perplexity = max(1, min(5, len(points) - 1))
    coords = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=42).fit_transform(matrix)

    plt.figure(figsize=(8, 6))
    labels = sorted({point.true_identity_id for point in points})
    cmap = plt.get_cmap("tab10", max(1, len(labels)))
    color_map = {label: cmap(index) for index, label in enumerate(labels)}

    for coord, point in zip(coords, points):
        is_failure = point.true_identity_id != point.predicted_identity_id
        plt.scatter(
            coord[0],
            coord[1],
            color=color_map[point.true_identity_id],
            marker="x" if is_failure else "o",
            s=120 if is_failure else 70,
            edgecolors="black" if not is_failure else None,
        )
        if is_failure:
            plt.annotate(
                f"{point.true_identity_id}->{point.predicted_identity_id}",
                (coord[0], coord[1]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color_map[label], markersize=8)
        for label in labels
    ]
    legend_handles.append(
        plt.Line2D([0], [0], marker="x", color="black", linestyle="None", label="misclassified", markersize=8)
    )
    plt.legend(handles=legend_handles)
    plt.title("Embedding t-SNE")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = create_model(args.model_name)
        points = load_points(args.report_path, model)
        write_tsne(points, args.output_path)
    except (FileNotFoundError, ImportError, InvalidImageError, RuntimeError, ValueError) as exc:
        print(f"t-SNE 可视化失败: {exc}")
        return 1

    print(f"t-SNE figure written to: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
