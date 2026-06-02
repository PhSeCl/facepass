import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


README_SECTION_START = "<!-- TEST_ANNOTATION_COUNTS:START -->"
README_SECTION_END = "<!-- TEST_ANNOTATION_COUNTS:END -->"
IDENTITY_PATTERN = re.compile(r"^p\d{2}$", re.IGNORECASE)


@dataclass(frozen=True)
class AnnotationSummary:
    annotation_path: Path
    images_dir: Path
    total_images: int
    total_faces: int
    unknown_faces: int
    identity_counts: dict[str, int]
    missing_annotation_images: list[str]
    dangling_annotation_images: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize dataset/test annotation identity counts and optionally refresh README."
    )
    parser.add_argument(
        "--annotation-path",
        default="dataset/test/annotation.json",
        help="Path to the grouped annotation JSON file.",
    )
    parser.add_argument(
        "--readme-path",
        default="README.md",
        help="README path used when --write-readme is enabled.",
    )
    parser.add_argument(
        "--images-dir",
        default="dataset/test/images",
        help="Image directory used to check whether every test image has a matching annotation entry.",
    )
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Replace the README annotation-count marker block with the generated markdown summary.",
    )
    return parser


def summarize_annotations(
    annotation_path: str | Path,
    images_dir: str | Path,
) -> AnnotationSummary:
    annotation_path = Path(annotation_path)
    images_dir = Path(images_dir)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"标注文件不是按图片分组的 JSON 对象: {annotation_path}")

    identity_counter: Counter[str] = Counter()
    total_faces = 0
    unknown_faces = 0

    for image_name, faces in payload.items():
        if not isinstance(image_name, str):
            raise ValueError(f"标注文件包含非法图片名: {image_name!r}")
        if not isinstance(faces, list):
            raise ValueError(f"图片 {image_name} 的标注不是列表")
        for face in faces:
            if not isinstance(face, dict):
                raise ValueError(f"图片 {image_name} 包含非法人脸标注项")
            identity = face.get("identity")
            if not isinstance(identity, str):
                raise ValueError(f"图片 {image_name} 存在缺失 identity 的标注项")
            total_faces += 1
            if identity.lower() == "unknown":
                unknown_faces += 1
                continue
            if not IDENTITY_PATTERN.fullmatch(identity):
                raise ValueError(f"图片 {image_name} 存在非法 identity: {identity}")
            identity_counter[identity.lower()] += 1

    annotated_images = sorted(str(image_name) for image_name in payload.keys())
    existing_images = sorted(path.name for path in images_dir.iterdir() if path.is_file())
    missing_annotation_images = sorted(set(existing_images) - set(annotated_images))
    dangling_annotation_images = sorted(set(annotated_images) - set(existing_images))

    return AnnotationSummary(
        annotation_path=annotation_path,
        images_dir=images_dir,
        total_images=len(payload),
        total_faces=total_faces,
        unknown_faces=unknown_faces,
        identity_counts=dict(sorted(identity_counter.items())),
        missing_annotation_images=missing_annotation_images,
        dangling_annotation_images=dangling_annotation_images,
    )


def render_markdown_summary(summary: AnnotationSummary) -> str:
    lines = [
        f"当前 [dataset/test/annotation.json]({summary.annotation_path.as_posix()}) 统计如下：",
        "",
        f"- 标注图片数：`{summary.total_images}`",
        f"- 标注人脸总数：`{summary.total_faces}`",
        f"- 其中 `unknown`：`{summary.unknown_faces}`",
        f"- 图片目录：`{summary.images_dir.as_posix()}`",
        f"- 缺少标注的图片：`{len(summary.missing_annotation_images)}`",
        f"- 多余标注项：`{len(summary.dangling_annotation_images)}`",
        "",
        "可用下面的命令重新生成本节：",
        "",
        "```powershell",
        "uv run python scripts/summarize_test_annotations.py --write-readme",
        "```",
        "",
        "如果上面两项不是 `0`，先修复 `dataset/test/annotation.json`，再继续评测或提交图片。",
        "",
        "| 身份 | 标注人数 |",
        "| --- | ---: |",
    ]
    for identity_id, count in summary.identity_counts.items():
        lines.append(f"| {identity_id} | {count} |")
    lines.append(f"| unknown | {summary.unknown_faces} |")
    return "\n".join(lines)


def update_readme_section(readme_path: str | Path, replacement: str) -> None:
    readme_path = Path(readme_path)
    content = readme_path.read_text(encoding="utf-8")
    start = content.find(README_SECTION_START)
    end = content.find(README_SECTION_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README 缺少标注统计标记块: {readme_path}")
    updated = (
        content[: start + len(README_SECTION_START)]
        + "\n"
        + replacement.rstrip()
        + "\n"
        + content[end:]
    )
    readme_path.write_text(updated, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_annotations(args.annotation_path, args.images_dir)
    markdown = render_markdown_summary(summary)
    if args.write_readme:
        update_readme_section(args.readme_path, markdown)
        print(f"README updated: {args.readme_path}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
