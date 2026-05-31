import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.errors import InvalidImageError
from src.eval.threshold import (
    SimilarityDistributions,
    collect_registered_embeddings,
    collect_similarity_distributions,
    suggest_threshold,
)
from src.face_model import create_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze registration-set similarity distributions to suggest an unknown threshold."
    )
    parser.add_argument(
        "--registered-root",
        default="data/registered",
        help="Root directory of registered identity images.",
    )
    parser.add_argument(
        "--histogram-path",
        default="reports/threshold_hist.png",
        help="Path for the histogram PNG output.",
    )
    parser.add_argument(
        "--model-name",
        default="insightface",
        help="Model factory name passed to create_model().",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help="Optional target false accept rate in [0, 1].",
    )
    return parser


def plot_histogram(
    distributions: SimilarityDistributions,
    histogram_path: Path,
    threshold: float,
) -> None:
    histogram_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.hist(distributions.same_identity, bins=20, alpha=0.6, label="same identity")
    plt.hist(distributions.different_identity, bins=20, alpha=0.6, label="different identity")
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4f}")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.title("Registration-set similarity distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = create_model(args.model_name)
        embeddings_by_identity = collect_registered_embeddings(args.registered_root, model)
        distributions = collect_similarity_distributions(embeddings_by_identity)
        threshold = suggest_threshold(distributions, target_false_accept_rate=args.target_far)
        plot_histogram(distributions, Path(args.histogram_path), threshold)
    except (FileNotFoundError, ImportError, InvalidImageError, RuntimeError, ValueError) as exc:
        print(f"阈值分析失败: {exc}")
        return 1

    print("threshold source: registration-set distributions only")
    print(f"suggested threshold: {threshold:.4f}")
    print(f"same-identity pairs: {len(distributions.same_identity)}")
    print(f"different-identity pairs: {len(distributions.different_identity)}")
    print(f"histogram written to: {args.histogram_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
