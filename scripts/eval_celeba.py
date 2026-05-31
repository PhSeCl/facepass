import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backend.gallery import Gallery
from src.common.errors import EmptyGalleryError, InvalidImageError
from src.common.images import safe_load_image
from src.common.logging import get_logger
from src.eval.celeba_dataset import CelebaDataset, load_celeba_dataset
from src.eval.evaluator import EvalReport, evaluate
from src.face_model import create_model


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate closed-set accuracy on CelebA identities.")
    parser.add_argument(
        "--dataset-root",
        default="celeba_100_identities_3reg_3test",
        help="Root directory containing register/ and test/ folders.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/celeba_eval.json",
        help="Path to the JSON report output.",
    )
    parser.add_argument(
        "--model-name",
        default="insightface",
        help="Model factory name passed to create_model().",
    )
    return parser


def build_gallery(register_root: Path, model) -> Gallery:
    gallery = Gallery()
    gallery.build_from_dir(str(register_root), model)
    return gallery


def evaluate_dataset(dataset: CelebaDataset, model) -> EvalReport:
    gallery = build_gallery(dataset.register_root, model)
    samples = [
        (safe_load_image(sample.path), sample.identity_id)
        for sample in dataset.test_samples
    ]
    return evaluate(model, gallery, samples)


def write_report(report_path: Path, dataset: CelebaDataset, report: EvalReport) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    per_class = {
        identity_id: {
            "correct": class_metrics.correct,
            "total": class_metrics.total,
            "accuracy": class_metrics.accuracy,
        }
        for identity_id, class_metrics in report.metrics.per_class.items()
    }
    samples = []
    success_samples = []
    failure_samples = []
    for dataset_sample, eval_sample in zip(dataset.test_samples, report.samples):
        item = {
            "path": str(dataset_sample.path),
            "true_identity_id": eval_sample.true_identity_id,
            "predicted_identity_id": eval_sample.predicted_identity_id,
            "similarity": eval_sample.similarity,
        }
        samples.append(item)
        if eval_sample.true_identity_id == eval_sample.predicted_identity_id:
            success_samples.append(item)
        else:
            failure_samples.append(item)

    payload = {
        "top1_accuracy": report.metrics.top1_accuracy,
        "correct": report.metrics.correct,
        "total": report.metrics.total,
        "per_class": per_class,
        "samples": samples,
        "success_samples": success_samples,
        "failure_samples": failure_samples,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root)
    report_path = Path(args.report_path)

    try:
        dataset = load_celeba_dataset(dataset_root)
        model = create_model(args.model_name)
        report = evaluate_dataset(dataset, model)
        write_report(report_path, dataset, report)
    except (EmptyGalleryError, FileNotFoundError, InvalidImageError, ImportError, RuntimeError, ValueError) as exc:
        print(f"CelebA 评测失败: {exc}")
        return 1

    print(f"top-1 accuracy: {report.metrics.top1_accuracy:.4f}")
    for identity_id, class_metrics in report.metrics.per_class.items():
        print(f"{identity_id}: {class_metrics.correct}/{class_metrics.total} = {class_metrics.accuracy:.4f}")
    print(f"report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
