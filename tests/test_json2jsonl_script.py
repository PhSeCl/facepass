import json
from pathlib import Path

from PIL import Image

from scripts.json2jsonl import (
    build_parser,
    convert_grouped,
    main,
    render_jsonl,
)


def write_image(images_dir: Path, name: str, size: tuple[int, int]) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(127, 127, 127)).save(images_dir / name)


def write_annotation(path: Path, payload: dict[str, list[dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_convert_renames_fields_and_sets_image_type(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    write_image(images_dir, "group_01.jpg", (200, 200))
    write_image(images_dir, "p02_t01.jpg", (200, 200))

    payload = {
        "group_01.jpg": [
            {"bbox": [10, 10, 30, 30], "identity": "p01", "score": 0.91},
            {"bbox": [60, 10, 30, 30], "identity": "unknown", "score": 0.08},
        ],
        "p02_t01.jpg": [
            {"bbox": [5, 5, 20, 20], "identity": "p02", "score": 0.88},
        ],
    }

    records = convert_grouped(payload, images_dir)

    assert records[0] == {
        "image": "test/images/group_01.jpg",
        "image_type": "multi",
        "faces": [
            {"identity_id": "p01", "bbox": [10, 10, 30, 30]},
            {"identity_id": "unknown", "bbox": [60, 10, 30, 30]},
        ],
    }
    # 单脸 -> single，且 score 被丢弃
    assert records[1]["image_type"] == "single"
    assert records[1]["faces"] == [{"identity_id": "p02", "bbox": [5, 5, 20, 20]}]


def test_convert_clamps_bbox_to_image_bounds(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    write_image(images_dir, "group_01.jpg", (100, 80))

    payload = {
        "group_01.jpg": [
            # 左/上越界：x、y 负值
            {"bbox": [-8, -5, 20, 30], "identity": "unknown", "score": 0.1},
            # 右/下越界：x+w、y+h 超出 100x80
            {"bbox": [90, 70, 40, 40], "identity": "p01", "score": 0.9},
        ],
    }

    faces = convert_grouped(payload, images_dir)[0]["faces"]

    assert faces[0]["bbox"] == [0, 0, 12, 25]   # x:-8->0 w:20-8=12 ; y:-5->0 h:30-5=25
    assert faces[1]["bbox"] == [90, 70, 10, 10]  # 右边 100-90=10 ; 下边 80-70=10


def test_convert_accepts_already_regulation_identity_id(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    write_image(images_dir, "p03_t01.jpg", (50, 50))

    records = convert_grouped(
        {"p03_t01.jpg": [{"bbox": [1, 1, 10, 10], "identity_id": "p03"}]},
        images_dir,
    )

    assert records[0]["faces"] == [{"identity_id": "p03", "bbox": [1, 1, 10, 10]}]


def test_render_jsonl_is_compact_and_newline_terminated() -> None:
    text = render_jsonl([{"image": "test/images/x.jpg", "image_type": "single", "faces": []}])

    assert text == '{"image":"test/images/x.jpg","image_type":"single","faces":[]}\n'


def test_main_writes_and_check_roundtrips(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.json"
    images_dir = tmp_path / "images"
    out_path = tmp_path / "annotations.jsonl"
    write_image(images_dir, "p01_t01.jpg", (60, 60))
    write_annotation(
        annotation_path,
        {"p01_t01.jpg": [{"bbox": [2, 2, 20, 20], "identity": "p01", "score": 0.9}]},
    )

    common = [
        "--annotation-path", str(annotation_path),
        "--images-dir", str(images_dir),
        "--out", str(out_path),
    ]

    assert main(common) == 0
    assert out_path.exists()
    # 刚生成完，--check 应判定为已同步
    assert main(common + ["--check"]) == 0

    # 改源文件后 --check 应判定为未同步
    write_annotation(
        annotation_path,
        {"p01_t01.jpg": [{"bbox": [3, 3, 20, 20], "identity": "p01", "score": 0.9}]},
    )
    assert main(common + ["--check"]) == 1


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.annotation_path == "dataset/test/annotation.json"
    assert args.out == "dataset/test/annotations.jsonl"
    assert args.images_dir == "dataset/test/images"
    assert args.check is False
