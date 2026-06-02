import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backend.config import settings
from src.backend.gallery import Gallery
from src.backend.recognizer import Recognizer
from src.common.errors import (
    EmptyGalleryError,
    InvalidImageError,
    ModelIncompleteError,
    ModelLoadError,
    ModelNotFoundError,
    ModelPathMissingError,
)
from src.eval.preannotation import generate_draft_annotations
from src.face_model import create_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate draft annotation.json for manual review from test images."
    )
    parser.add_argument(
        "--images-dir",
        default="dataset/test/images",
        help="Directory containing test images to pre-annotate.",
    )
    parser.add_argument(
        "--out",
        default="dataset/test/annotation.json",
        help="Path to the generated annotation JSON file.",
    )
    parser.add_argument(
        "--registered-root",
        default="dataset/registered",
        help="Root directory for registered identity images.",
    )
    parser.add_argument(
        "--gallery-path",
        default=str(settings.gallery_path),
        help="Path to gallery cache used by the backend startup flow.",
    )
    parser.add_argument(
        "--model-name",
        default=settings.model_name,
        help="Model factory name passed to create_model().",
    )
    parser.add_argument(
        "--model-path",
        help="Optional local model directory passed through when using insightface.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.45,
        help="Confidence line used only to flag draft labels for manual review.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing annotation file.",
    )
    return parser


def build_recognizer(
    registered_root: str | Path,
    gallery_path: str | Path,
    model_name: str,
    model_path: str | Path | None = None,
    threshold: float = settings.threshold,
) -> Recognizer:
    model_kwargs = {}
    if model_name.lower() == "insightface":
        model_kwargs["model_path"] = model_path
    model = create_model(model_name, **model_kwargs)

    gallery_path = Path(gallery_path)
    if gallery_path.exists():
        gallery = Gallery.load(gallery_path)
    else:
        gallery = Gallery()
        gallery.build_from_dir(str(registered_root), model)
        gallery.save(gallery_path)
    return Recognizer(model=model, gallery=gallery, threshold=threshold, id2name={})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.out)
    if output_path.exists() and not args.overwrite:
        print(f"预标注已存在，默认不覆盖: {output_path}。如需重写请添加 --overwrite。")
        return 1

    try:
        recognizer = build_recognizer(
            registered_root=args.registered_root,
            gallery_path=args.gallery_path,
            model_name=args.model_name,
            model_path=args.model_path,
        )
        summary = generate_draft_annotations(
            recognizer=recognizer,
            images_dir=args.images_dir,
            out_path=output_path,
            review_threshold=args.review_threshold,
            overwrite=args.overwrite,
        )
    except (
        EmptyGalleryError,
        FileExistsError,
        FileNotFoundError,
        ImportError,
        InvalidImageError,
        ModelIncompleteError,
        ModelLoadError,
        ModelNotFoundError,
        ModelPathMissingError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"预标注失败: {exc}")
        return 1

    print(
        "预标注完成: "
        f"images={summary.processed_images}, faces={summary.total_faces}, review={summary.review_images}"
    )
    print(f"annotation written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
