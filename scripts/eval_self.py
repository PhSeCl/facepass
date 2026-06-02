import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backend.config import settings
from src.backend.gallery import Gallery
from src.common.errors import EmptyGalleryError, InvalidImageError
from src.eval.evaluator import evaluate
from src.eval.metrics import compute_accuracy_metrics
from src.eval.self_dataset import SelfDataset, load_self_dataset
from src.face_model import create_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate self-collected identities and unknown faces.")
    parser.add_argument(
        "--annotations-path",
        default="dataset/test/annotations.jsonl",
        help="JSONL file containing image_path and faces annotations.",
    )
    parser.add_argument(
        "--test-root",
        default="dataset/test",
        help="Root directory for annotated test images.",
    )
    parser.add_argument(
        "--registered-root",
        default="dataset/registered",
        help="Root directory for registered identity images.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/self_eval.json",
        help="Path to the JSON report output.",
    )
    parser.add_argument(
        "--model-name",
        default="insightface",
        help="Model factory name passed to create_model().",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.threshold,
        help="Similarity threshold used to mark predictions as unknown.",
    )
    return parser


def build_gallery(registered_root: Path, model) -> Gallery:
    gallery = Gallery()
    gallery.build_from_dir(str(registered_root), model)
    return gallery


def evaluate_dataset(dataset: SelfDataset, model, threshold: float):
    gallery = build_gallery(dataset.registered_root, model)
    report = evaluate(
        model,
        gallery,
        [(sample.face_image, sample.identity_id) for sample in dataset.samples],
        threshold=threshold,
    )

    known_pairs = [
        (sample.true_identity_id, sample.predicted_identity_id)
        for sample in report.samples
        if sample.true_identity_id != "unknown"
    ]
    known_metrics = compute_accuracy_metrics(known_pairs)

    unknown_samples = [sample for sample in report.samples if sample.true_identity_id == "unknown"]
    unknown_total = len(unknown_samples)
    unknown_predicted_unknown = sum(
        1 for sample in unknown_samples if sample.predicted_identity_id == "unknown"
    )
    unknown_accuracy = (
        unknown_predicted_unknown / unknown_total if unknown_total else 0.0
    )
    return report, known_metrics, unknown_total, unknown_predicted_unknown, unknown_accuracy


def write_report(
    report_path: Path,
    dataset: SelfDataset,
    report,
    known_metrics,
    unknown_total: int,
    unknown_predicted_unknown: int,
    unknown_accuracy: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    per_class = {
        identity_id: {
            "correct": class_metrics.correct,
            "total": class_metrics.total,
            "accuracy": class_metrics.accuracy,
        }
        for identity_id, class_metrics in known_metrics.per_class.items()
    }
    samples = []
    for dataset_sample, eval_sample in zip(dataset.samples, report.samples):
        samples.append(
            {
                "path": str(dataset_sample.source_path),
                "bbox": list(dataset_sample.bbox),
                "true_identity_id": eval_sample.true_identity_id,
                "predicted_identity_id": eval_sample.predicted_identity_id,
                "similarity": eval_sample.similarity,
            }
        )

    payload = {
        "known_top1_accuracy": known_metrics.top1_accuracy,
        "known_correct": known_metrics.correct,
        "known_total": known_metrics.total,
        "per_class": per_class,
        "unknown_total": unknown_total,
        "unknown_predicted_unknown": unknown_predicted_unknown,
        "unknown_accuracy": unknown_accuracy,
        "samples": samples,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = load_self_dataset(args.annotations_path, args.test_root, args.registered_root)
        model = create_model(args.model_name)
        report, known_metrics, unknown_total, unknown_predicted_unknown, unknown_accuracy = evaluate_dataset(
            dataset, model, args.threshold
        )
        write_report(
            Path(args.report_path),
            dataset,
            report,
            known_metrics,
            unknown_total,
            unknown_predicted_unknown,
            unknown_accuracy,
        )
    except (EmptyGalleryError, FileNotFoundError, InvalidImageError, ImportError, RuntimeError, ValueError) as exc:
        print(f"自采评测失败: {exc}")
        return 1

    print(f"known top-1 accuracy: {known_metrics.top1_accuracy:.4f}")
    for identity_id, class_metrics in known_metrics.per_class.items():
        print(f"{identity_id}: {class_metrics.correct}/{class_metrics.total} = {class_metrics.accuracy:.4f}")
    print(
        "unknown accuracy: "
        f"{unknown_accuracy:.4f} ({unknown_predicted_unknown}/{unknown_total})"
    )
    print(f"report written to: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
