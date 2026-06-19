"""把人工维护的分组式 annotation.json 转换为作业规范要求的 annotations.jsonl。

分组式 `dataset/test/annotation.json` 可读性好、便于人工核对，是日常维护的主文件；
而 `dataset/test/annotations.jsonl` 是严格对齐作业规范（逐行、字段为
image/image_type/faces[].identity_id+bbox）的版本。本脚本由前者生成后者，保证两份
文件始终一致——新增图片、改完 annotation.json 后跑一次即可，CI 也会自动同步。

用法：
    uv run python scripts/json2jsonl.py            # 重新生成 annotations.jsonl
    uv run python scripts/json2jsonl.py --check     # 只校验是否已同步（CI 用，不写文件）
"""

import argparse
import json
from pathlib import Path
import re

from PIL import Image


# 合法 identity：p 开头 + 两位数字，或 unknown。与 src/eval 加载器的校验口径一致；
# 这里内联而非 import，是为了让脚本只依赖 Pillow + 标准库，避免在仅装了转换依赖的
# CI 环境里因 src.eval 包链（numpy 等）导入失败。
IDENTITY_PATTERN = re.compile(r"p\d{2}|unknown")

# annotations.jsonl 中 image 字段相对 dataset 根目录的固定前缀。
IMAGE_PREFIX = "test/images"


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    """把 [x, y, w, h] 钳制到图像边界 [0, 0, width, height] 内。

    检测器产出的原始框可能略微越界（负坐标或超出右/下边缘），但作业规范要求
    bbox 为图像内的绝对像素坐标，因此这里裁掉越界部分。
    """
    x, y, w, h = bbox
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    return [x0, y0, x1 - x0, y1 - y0]


def _convert_face(face: object, source_label: str, width: int, height: int) -> dict[str, object]:
    if not isinstance(face, dict):
        raise ValueError(f"{source_label}: 人脸标注必须是对象")

    bbox = face.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{source_label}: bbox 必须是 [x, y, w, h] 四元组")

    # 分组式用 identity，规范式用 identity_id；两者都兼容。
    identity_value = face.get("identity_id", face.get("identity"))
    if identity_value is None:
        raise ValueError(f"{source_label}: 缺少 identity / identity_id")
    identity_id = str(identity_value)
    if not IDENTITY_PATTERN.fullmatch(identity_id):
        raise ValueError(f"{source_label}: 非法 identity {identity_id!r}（应为 p01..p20 或 unknown）")

    # 丢弃 score 等预标注辅助字段，bbox 取整后钳制到图像边界。
    clamped = _clamp_bbox([int(value) for value in bbox], width, height)
    return {"identity_id": identity_id, "bbox": clamped}


def convert_grouped(payload: dict[str, object], images_dir: Path) -> list[dict[str, object]]:
    """把分组式标注 dict 转换为规范式记录列表，保留原插入顺序。

    需要 images_dir 读取每张图片的真实尺寸，以便把 bbox 钳制到图像边界内。
    """
    if not isinstance(payload, dict):
        raise ValueError("annotation.json 顶层必须是按图片名分组的对象")

    records: list[dict[str, object]] = []
    for image_name, faces in payload.items():
        if not isinstance(faces, list):
            raise ValueError(f"{image_name}: 标注必须是人脸列表")
        image_file = images_dir / image_name
        if not image_file.exists():
            raise FileNotFoundError(f"标注引用的图片不存在: {image_file}")
        with Image.open(image_file) as image:
            width, height = image.size
        converted_faces = [
            _convert_face(face, f"{image_name} 第 {index} 个 face", width, height)
            for index, face in enumerate(faces, start=1)
        ]
        # image_type 按人脸数判定：多于一张为 multi，否则 single。
        image_type = "multi" if len(converted_faces) > 1 else "single"
        records.append(
            {
                "image": f"{IMAGE_PREFIX}/{image_name}",
                "image_type": image_type,
                "faces": converted_faces,
            }
        )
    return records


def render_jsonl(records: list[dict[str, object]]) -> str:
    """渲染为紧凑 JSONL 文本（每行一条记录，末尾带换行）。"""
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
    return "".join(f"{line}\n" for line in lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把分组式 annotation.json 转换为规范式 annotations.jsonl。"
    )
    parser.add_argument(
        "--annotation-path",
        default="dataset/test/annotation.json",
        help="分组式标注源文件路径。",
    )
    parser.add_argument(
        "--out",
        default="dataset/test/annotations.jsonl",
        help="规范式 JSONL 输出路径。",
    )
    parser.add_argument(
        "--images-dir",
        default="dataset/test/images",
        help="测试图片目录，用于读取尺寸并把 bbox 钳制到图像边界。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验输出是否已与源文件同步，不写文件；不一致时返回非零退出码。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    annotation_path = Path(args.annotation_path)
    out_path = Path(args.out)

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    rendered = render_jsonl(convert_grouped(payload, Path(args.images_dir)))

    if args.check:
        current = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        if current == rendered:
            print(f"已同步: {out_path}")
            return 0
        print(
            f"未同步: {out_path} 与 {annotation_path} 不一致，"
            f"请运行 `uv run python scripts/json2jsonl.py` 重新生成。"
        )
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"已写入 {out_path}（{rendered.count(chr(10))} 条记录），源: {annotation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
