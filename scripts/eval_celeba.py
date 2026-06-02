import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backend.gallery import Gallery
from src.common.errors import EmptyGalleryError, InvalidImageError
from src.eval.celeba_dataset import CelebADataset, load_celeba_dataset
from src.eval.classification_reporting import plot_per_class_accuracy, plot_top1_accuracy
from src.eval.evaluator import evaluate
from src.face_model import create_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CelebA 100-class cropped-face recognition.")
    parser.add_argument(
        "--data-dir",
        default="celeba_100_identities_3reg_3test",
        help="CelebA root directory containing register/ and test/.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/celeba_eval.json",
        help="Path to the JSON report output.",
    )
    parser.add_argument(
        "--top1-plot-path",
        default="reports/celeba_top1_accuracy.png",
        help="Path to the top-1 accuracy PNG output.",
    )
    parser.add_argument(
        "--per-class-plot-path",
        default="reports/celeba_per_class_accuracy.png",
        help="Path to the per-class accuracy PNG output.",
    )
    parser.add_argument(
        "--model-name",
        default="insightface",
        help="Model factory name passed to create_model().",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Maximum number of success/failure samples included in console output and report summaries.",
    )
    return parser


def build_gallery(dataset: CelebADataset, model) -> Gallery:
    gallery = Gallery()
    gallery.build_from_cropped_dir(str(dataset.register_dir), model)
    return gallery


def _summarize_samples(dataset: CelebADataset, report, sample_limit: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for dataset_sample, eval_sample in zip(dataset.samples, report.samples):
        item = {
            "path": str(dataset_sample.source_path),
            "true_identity_id": eval_sample.true_identity_id,
            "predicted_identity_id": eval_sample.predicted_identity_id,
            "similarity": eval_sample.similarity,
        }
        if eval_sample.true_identity_id == eval_sample.predicted_identity_id:
            if len(successes) < sample_limit:
                successes.append(item)
        else:
            if len(failures) < sample_limit:
                failures.append(item)
    return successes, failures


def write_report(report_path: Path, dataset: CelebADataset, report, sample_limit: int) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    success_samples, failure_samples = _summarize_samples(dataset, report, sample_limit)
    per_class = {
        identity_id: {
            "correct": class_metrics.correct,
            "total": class_metrics.total,
            "accuracy": class_metrics.accuracy,
        }
        for identity_id, class_metrics in report.metrics.per_class.items()
    }
    payload = {
        "top1_accuracy": report.metrics.top1_accuracy,
        "correct": report.metrics.correct,
        "total": report.metrics.total,
        "per_class": per_class,
        "success_samples": success_samples,
        "failure_samples": failure_samples,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_plots(report, top1_plot_path: Path, per_class_plot_path: Path) -> None:
    plot_top1_accuracy(report.metrics, top1_plot_path)
    plot_per_class_accuracy(report.metrics, per_class_plot_path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = load_celeba_dataset(args.data_dir)
        model = create_model(args.model_name)
        gallery = build_gallery(dataset, model)
        report = evaluate(
            model,
            gallery,
            [(sample.face_image, sample.identity_id) for sample in dataset.samples],
            threshold=None,
        )
        report_path = Path(args.report_path)
        write_report(report_path, dataset, report, args.sample_limit)
        write_plots(report, Path(args.top1_plot_path), Path(args.per_class_plot_path))
    except (EmptyGalleryError, FileNotFoundError, InvalidImageError, ImportError, RuntimeError, ValueError) as exc:
        print(f"CelebA 评测失败: {exc}")
        return 1

    success_samples, failure_samples = _summarize_samples(dataset, report, args.sample_limit)
    print(f"top-1 accuracy: {report.metrics.top1_accuracy:.4f}")
    print(f"correct/total: {report.metrics.correct}/{report.metrics.total}")
    if success_samples:
        print("success samples:")
        for item in success_samples:
            print(
                f"- {item['path']}: "
                f"{item['true_identity_id']} -> {item['predicted_identity_id']} "
                f"(sim={item['similarity']:.4f})"
            )
    if failure_samples:
        print("failure samples:")
        for item in failure_samples:
            print(
                f"- {item['path']}: "
                f"{item['true_identity_id']} -> {item['predicted_identity_id']} "
                f"(sim={item['similarity']:.4f})"
            )
    print(f"report written to: {args.report_path}")
    print(f"top-1 plot written to: {args.top1_plot_path}")
    print(f"per-class plot written to: {args.per_class_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
