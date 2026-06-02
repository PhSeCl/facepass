import json
from pathlib import Path

from scripts.summarize_test_annotations import (
    build_parser,
    render_markdown_summary,
    summarize_annotations,
    update_readme_section,
)


def write_annotation(path: Path, payload: dict[str, list[dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_summarize_annotations_counts_identities_and_unknowns(tmp_path: Path) -> None:
    annotation_path = tmp_path / "dataset" / "test" / "annotation.json"
    images_dir = annotation_path.parent / "images"
    images_dir.mkdir(parents=True)
    for image_name in ("group_01.jpg", "p02_t01.jpg", "p02_t02.jpg", "p03_t03.jpg"):
        (images_dir / image_name).write_bytes(b"fake")
    write_annotation(
        annotation_path,
        {
            "group_01.jpg": [
                {"bbox": [0, 0, 10, 10], "identity": "p01", "score": 0.91},
                {"bbox": [10, 0, 10, 10], "identity": "unknown", "score": 0.08},
            ],
            "p02_t01.jpg": [
                {"bbox": [1, 1, 8, 8], "identity": "p02", "score": 0.88},
            ],
            "p02_t02.jpg": [
                {"bbox": [2, 2, 8, 8], "identity": "p02", "score": 0.93},
            ],
            "ghost.jpg": [
                {"bbox": [2, 2, 8, 8], "identity": "unknown", "score": 0.12},
            ],
        },
    )

    summary = summarize_annotations(annotation_path, images_dir)

    assert summary.total_images == 4
    assert summary.total_faces == 5
    assert summary.unknown_faces == 2
    assert summary.identity_counts == {"p01": 1, "p02": 2}
    assert summary.missing_annotation_images == ["p03_t03.jpg"]
    assert summary.dangling_annotation_images == ["ghost.jpg"]


def test_render_markdown_summary_outputs_table_and_command() -> None:
    from scripts.summarize_test_annotations import AnnotationSummary

    summary = AnnotationSummary(
        annotation_path=Path("dataset/test/annotation.json"),
        images_dir=Path("dataset/test/images"),
        total_images=3,
        total_faces=4,
        unknown_faces=1,
        identity_counts={"p01": 1, "p02": 2},
        missing_annotation_images=[],
        dangling_annotation_images=[],
    )

    markdown = render_markdown_summary(summary)

    assert "dataset/test/annotation.json" in markdown
    assert "| 身份 | 标注人数 |" in markdown
    assert "| p01 | 1 |" in markdown
    assert "| p02 | 2 |" in markdown
    assert "| unknown | 1 |" in markdown


def test_update_readme_section_replaces_marker_block(tmp_path: Path) -> None:
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Title",
                "<!-- TEST_ANNOTATION_COUNTS:START -->",
                "old content",
                "<!-- TEST_ANNOTATION_COUNTS:END -->",
            ]
        ),
        encoding="utf-8",
    )

    update_readme_section(readme_path, "new summary")

    content = readme_path.read_text(encoding="utf-8")
    assert "new summary" in content
    assert "old content" not in content


def test_summary_parser_defaults_to_dataset_annotation_and_readme() -> None:
    args = build_parser().parse_args([])

    assert args.annotation_path == "dataset/test/annotation.json"
    assert args.images_dir == "dataset/test/images"
    assert args.readme_path == "README.md"
    assert args.write_readme is False
