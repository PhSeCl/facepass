import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import gradio as gr
import requests
from PIL import Image, ImageDraw


BACKEND_URL = os.getenv("FACEPASS_BACKEND_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 5
DATASET_EVAL_TIMEOUT = 60
_dataset_browser_root = Path(os.getenv("FACEPASS_FRONTEND_FILE_ROOT", Path.cwd().anchor or str(Path.cwd())))
DATASET_BROWSER_ROOT = str(_dataset_browser_root if _dataset_browser_root.exists() else Path.cwd())
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
def _post_dataset_inspect_directory(path: str, backend_url: str) -> requests.Response:
    return requests.post(
        f"{backend_url.rstrip('/')}/dataset-eval/inspect",
        data={"dataset_dir": path},
        timeout=REQUEST_TIMEOUT,
    )


def _post_dataset_eval(path: str, gallery_choice: str, backend_url: str) -> requests.Response:
    with open(path, "rb") as handle:
        return requests.post(
            f"{backend_url.rstrip('/')}/dataset-eval/run",
            data={"gallery_choice": gallery_choice},
            files={"file": (Path(path).name, handle, "application/zip")},
            timeout=DATASET_EVAL_TIMEOUT,
        )


def _post_dataset_eval_directory(path: str, gallery_choice: str, backend_url: str) -> requests.Response:
    return requests.post(
        f"{backend_url.rstrip('/')}/dataset-eval/run",
        data={"gallery_choice": gallery_choice, "dataset_dir": path},
        timeout=DATASET_EVAL_TIMEOUT,
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


def _render_plot_html(data_url: str, alt: str) -> str:
    return (
        f'<div style="padding: 8px 0;">'
        f'<img src="{data_url}" alt="{alt}" style="max-width: 100%; height: auto; display: block;" />'
        f"</div>"
    )


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


def inspect_dataset_via_backend(
    dataset_path: str | None,
    source_mode: str = "zip",
    backend_url: str = BACKEND_URL,
) -> tuple[bool, str]:
    if not dataset_path:
        if source_mode == "directory":
            return False, "请先选择一个数据集文件夹"
        return False, "请先上传一个 test.zip"
    if source_mode == "directory" and not Path(dataset_path).is_dir():
        return False, "请选择文件夹，不要选择具体文件"
    try:
        if source_mode == "directory":
            response = _post_dataset_inspect_directory(dataset_path, backend_url)
        else:
            response = _post_dataset_inspect(dataset_path, backend_url)
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
        if source_mode == "directory":
            return True, "检测到文件夹内含 `registered/`，请选择使用本组底库还是文件夹内底库。"
        return True, "检测到 zip 内含 `registered/`，请选择使用本组底库还是 zip 内底库。"
    if source_mode == "directory":
        return False, "文件夹内未检测到 `registered/`，将直接使用本组底库。"
    return False, "zip 内未检测到 `registered/`，将直接使用本组底库。"


def run_dataset_eval_via_backend(
    dataset_path: str | None,
    gallery_choice: str,
    source_mode: str = "zip",
    backend_url: str = BACKEND_URL,
):
    if not dataset_path:
        if source_mode == "directory":
            return "", "", "", "", [], [], "请先选择一个数据集文件夹"
        return "", "", "", "", [], [], "请先上传一个 test.zip"
    if source_mode == "directory" and not Path(dataset_path).is_dir():
        return "", "", "", "", [], [], "请选择文件夹，不要选择具体文件"
    try:
        if source_mode == "directory":
            response = _post_dataset_eval_directory(dataset_path, gallery_choice, backend_url)
        else:
            response = _post_dataset_eval(dataset_path, gallery_choice, backend_url)
    except requests.Timeout:
        return "", "", "", "", [], [], "数据集评测超时，请稍后重试或检查后端日志"
    except requests.ConnectionError:
        return "", "", "", "", [], [], "后端未启动，请先按 README 启动 uvicorn 服务"

    if response.status_code >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        return "", "", "", "", [], [], _friendly_http_error(response.status_code, payload)

    payload = response.json()
    plots = payload["plots"]
    return (
        _format_eval_summary(payload),
        _render_plot_html(plots["confusion_matrix"], "混淆矩阵"),
        _render_plot_html(plots["detection_metrics"], "检测指标"),
        _render_plot_html(plots["accuracy_metrics"], "识别准确率"),
        _format_bbox_rows(payload.get("missed_detections", [])),
        _format_bbox_rows(payload.get("false_positives", [])),
        "",
    )


def _selected_dataset_path(source_mode: str, archive_path: str | None, directory_path: str | None) -> str | None:
    if source_mode == "directory":
        return directory_path
    return archive_path


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
            dataset_source = gr.Radio(
                choices=[("ZIP", "zip"), ("文件夹", "directory")],
                value="zip",
                label="输入来源",
            )
            archive_input = gr.File(label="上传 test.zip", file_types=[".zip"], type="filepath", visible=True)
            directory_input = gr.FileExplorer(
                label="选择数据集文件夹",
                root_dir=DATASET_BROWSER_ROOT,
                file_count="single",
                visible=False,
                height=320,
            )
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
            confusion_output = gr.HTML(label="混淆矩阵")
            detection_output = gr.HTML(label="检测指标")
            accuracy_output = gr.HTML(label="识别准确率")
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

            def on_dataset_source_change(source_mode: str):
                return (
                    gr.update(visible=source_mode == "zip", value=None),
                    gr.update(visible=source_mode == "directory", value=None),
                    gr.update(visible=False, value="local"),
                    "",
                )

            def inspect_dataset_ui(source_mode: str, archive_path: str | None, directory_path: str | None):
                has_registered, info = inspect_dataset_via_backend(
                    _selected_dataset_path(source_mode, archive_path, directory_path),
                    source_mode=source_mode,
                )
                return gr.update(visible=has_registered, value="local"), info

            def run_dataset_eval_ui(
                source_mode: str,
                archive_path: str | None,
                directory_path: str | None,
                gallery_choice: str,
            ):
                return run_dataset_eval_via_backend(
                    _selected_dataset_path(source_mode, archive_path, directory_path),
                    gallery_choice,
                    source_mode=source_mode,
                )

            dataset_source.change(
                on_dataset_source_change,
                inputs=dataset_source,
                outputs=[archive_input, directory_input, gallery_choice, dataset_message],
            )
            inspect_button.click(
                inspect_dataset_ui,
                inputs=[dataset_source, archive_input, directory_input],
                outputs=[gallery_choice, dataset_message],
            )
            archive_input.change(
                inspect_dataset_ui,
                inputs=[dataset_source, archive_input, directory_input],
                outputs=[gallery_choice, dataset_message],
            )
            directory_input.change(
                inspect_dataset_ui,
                inputs=[dataset_source, archive_input, directory_input],
                outputs=[gallery_choice, dataset_message],
            )
            evaluate_button.click(
                run_dataset_eval_ui,
                inputs=[dataset_source, archive_input, directory_input, gallery_choice],
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
