import os
import time
import base64
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import ParamSpec, TypeVar

import gradio as gr
import requests
from PIL import Image, ImageDraw


BACKEND_URL = os.getenv("FACEPASS_BACKEND_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 5
P = ParamSpec("P")
R = TypeVar("R")


def _with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = (requests.ConnectionError, requests.Timeout),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_error: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_error = exc
                    if attempt == max_attempts:
                        break
                    time.sleep(base_delay * (2 ** (attempt - 1)))
            assert last_error is not None
            raise last_error

        return wrapper

    return decorator


def _friendly_http_error(status_code: int, payload: dict | None = None) -> str:
    message = ""
    if payload:
        detail = payload.get("detail", payload)
        if isinstance(detail, dict):
            message = str(detail.get("message", ""))
        else:
            message = str(detail)
    if status_code == 400:
        return message or "图片格式不支持，请换一张"
    if status_code == 413:
        return message or "图片过大，请换一张较小的图片"
    return message or f"后端返回错误 {status_code}，请查看后端日志"


@_with_retry(
    max_attempts=3,
    base_delay=0.5,
    exceptions=(requests.ConnectionError, requests.Timeout),
)
def _post_recognize(path: str, backend_url: str) -> requests.Response:
    with open(path, "rb") as handle:
        return requests.post(
            f"{backend_url.rstrip('/')}/recognize",
            files={"file": (Path(path).name, handle, "application/octet-stream")},
            timeout=REQUEST_TIMEOUT,
        )


@_with_retry(
    max_attempts=3,
    base_delay=0.5,
    exceptions=(requests.ConnectionError, requests.Timeout),
)
def _post_dataset_inspect(path: str, backend_url: str) -> requests.Response:
    with open(path, "rb") as handle:
        return requests.post(
            f"{backend_url.rstrip('/')}/dataset-eval/inspect",
            files={"file": (Path(path).name, handle, "application/zip")},
            timeout=REQUEST_TIMEOUT,
        )


@_with_retry(
    max_attempts=3,
    base_delay=0.5,
    exceptions=(requests.ConnectionError, requests.Timeout),
)
def _post_dataset_eval(path: str, gallery_choice: str, backend_url: str) -> requests.Response:
    with open(path, "rb") as handle:
        return requests.post(
            f"{backend_url.rstrip('/')}/dataset-eval/run",
            data={"gallery_choice": gallery_choice},
            files={"file": (Path(path).name, handle, "application/zip")},
            timeout=REQUEST_TIMEOUT,
        )


def _draw_results(image_path: str, results: list[dict]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for result in results:
        x, y, w, h = result["bbox"]
        color = (220, 38, 38) if result["is_unknown"] else (22, 163, 74)
        label_name = result["identity_id"] if result["is_unknown"] else result.get("name") or result["identity_id"]
        label = f"{label_name} {result['similarity']:.2f}"
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        draw.text((x, max(0, y - 14)), label, fill=color)
    return image


def _decode_plot_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _format_bbox_rows(items: list[dict]) -> list[list[str]]:
    return [[item["image_name"], ", ".join(str(value) for value in item["bbox"])] for item in items]


def _format_eval_summary(payload: dict) -> str:
    metrics = payload["metrics"]
    gallery_source = payload.get("gallery_source", "local")
    return (
        f"**底库来源**: `{gallery_source}`\n\n"
        f"- strict top-1: `{metrics['strict_top1_accuracy']:.3f}`\n"
        f"- matched top-1: `{metrics['matched_top1_accuracy']:.3f}`\n"
        f"- detection recall: `{metrics['detection_recall']:.3f}`\n"
        f"- detection precision: `{metrics['detection_precision']:.3f}`\n"
        f"- unknown detected accuracy: `{metrics['unknown_detected_accuracy']:.3f}`\n"
        f"- predicted unknown precision: `{metrics['predicted_unknown_precision']:.3f}`"
    )


def recognize_via_backend(image_path: str | None, backend_url: str = BACKEND_URL):
    if not image_path:
        return None, [], "请先上传一张图片"
    try:
        Image.open(image_path).verify()
    except Exception:
        return None, [], "图片格式不支持，请换一张"

    try:
        response = _post_recognize(image_path, backend_url)
    except (requests.ConnectionError, requests.Timeout):
        return None, [], "后端未启动，请先按 README 启动 uvicorn 服务"

    if getattr(response, "status_code", 200) >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        return None, [], _friendly_http_error(response.status_code, payload)

    results = response.json()
    if not results:
        return Image.open(image_path).convert("RGB"), [], "未在图中检测到人脸"

    rows = [
        [
            item["identity_id"],
            item.get("name") or "",
            f"{item['similarity']:.3f}",
            "是" if item["is_unknown"] else "否",
        ]
        for item in results
    ]
    return _draw_results(image_path, results), rows, ""


def inspect_dataset_via_backend(archive_path: str | None, backend_url: str = BACKEND_URL) -> tuple[bool, str]:
    if not archive_path:
        return False, "请先上传一个 test.zip"
    try:
        response = _post_dataset_inspect(archive_path, backend_url)
    except (requests.ConnectionError, requests.Timeout):
        return False, "后端未启动，请先按 README 启动 uvicorn 服务"

    if getattr(response, "status_code", 200) >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        return False, _friendly_http_error(response.status_code, payload)

    has_registered = bool(response.json().get("has_registered"))
    if has_registered:
        return True, "检测到 zip 内含 `registered/`，请选择使用本组底库还是 zip 内底库。"
    return False, "zip 内未检测到 `registered/`，将直接使用本组底库。"


def run_dataset_eval_via_backend(
    archive_path: str | None,
    gallery_choice: str,
    backend_url: str = BACKEND_URL,
):
    if not archive_path:
        return "", None, None, None, [], [], "请先上传一个 test.zip"
    try:
        response = _post_dataset_eval(archive_path, gallery_choice, backend_url)
    except (requests.ConnectionError, requests.Timeout):
        return "", None, None, None, [], [], "后端未启动，请先按 README 启动 uvicorn 服务"

    if response.status_code >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        return "", None, None, None, [], [], _friendly_http_error(response.status_code, payload)

    payload = response.json()
    plots = payload["plots"]
    return (
        _format_eval_summary(payload),
        _decode_plot_image(plots["confusion_matrix"]),
        _decode_plot_image(plots["detection_metrics"]),
        _decode_plot_image(plots["accuracy_metrics"]),
        _format_bbox_rows(payload.get("missed_detections", [])),
        _format_bbox_rows(payload.get("false_positives", [])),
        "",
    )


def load_identities(backend_url: str = BACKEND_URL):
    try:
        response = requests.get(f"{backend_url.rstrip('/')}/identities", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return [
            [item["identity_id"], item.get("name") or "", item["count"]]
            for item in response.json().get("identities", [])
        ]
    except requests.RequestException:
        return []


with gr.Blocks(title="FacePass") as demo:
    with gr.Tab("演示"):
        with gr.Tab("单图演示"):
            image_input = gr.Image(type="filepath", label="上传图片")
            run_button = gr.Button("识别", variant="primary")
            message = gr.Markdown()
            annotated_output = gr.Image(label="识别结果")
            table_output = gr.Dataframe(
                headers=["身份", "姓名", "相似度", "是否 unknown"],
                datatype=["str", "str", "str", "str"],
                label="结果表格",
            )
            run_button.click(
                recognize_via_backend,
                inputs=image_input,
                outputs=[annotated_output, table_output, message],
            )
        with gr.Tab("数据集演示"):
            archive_input = gr.File(label="上传 test.zip", file_types=[".zip"], type="filepath")
            inspect_button = gr.Button("检查底库来源", variant="secondary")
            gallery_choice = gr.Radio(
                choices=[("本组底库", "local"), ("zip 内底库", "archive")],
                value="local",
                label="底库来源",
                visible=False,
            )
            evaluate_button = gr.Button("运行数据集评测", variant="primary")
            dataset_message = gr.Markdown()
            summary_output = gr.Markdown(label="评测摘要")
            confusion_output = gr.Image(label="混淆矩阵")
            detection_output = gr.Image(label="检测指标")
            accuracy_output = gr.Image(label="识别准确率")
            missed_output = gr.Dataframe(
                headers=["图片", "bbox"],
                datatype=["str", "str"],
                label="漏检明细",
            )
            false_output = gr.Dataframe(
                headers=["图片", "bbox"],
                datatype=["str", "str"],
                label="误检明细",
            )

            def inspect_dataset_ui(archive_path: str | None):
                has_registered, info = inspect_dataset_via_backend(archive_path)
                return gr.update(visible=has_registered, value="local"), info

            inspect_button.click(
                inspect_dataset_ui,
                inputs=archive_input,
                outputs=[gallery_choice, dataset_message],
            )
            archive_input.change(
                inspect_dataset_ui,
                inputs=archive_input,
                outputs=[gallery_choice, dataset_message],
            )
            evaluate_button.click(
                run_dataset_eval_via_backend,
                inputs=[archive_input, gallery_choice],
                outputs=[
                    summary_output,
                    confusion_output,
                    detection_output,
                    accuracy_output,
                    missed_output,
                    false_output,
                    dataset_message,
                ],
            )
    with gr.Tab("身份库"):
        refresh = gr.Button("刷新身份库")
        identities_table = gr.Dataframe(
            headers=["身份", "姓名", "注册图数量"],
            datatype=["str", "str", "number"],
        )
        refresh.click(load_identities, outputs=identities_table)


if __name__ == "__main__":
    demo.launch()
