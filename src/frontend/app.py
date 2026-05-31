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

    if response.status_code >= 400:
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
    with gr.Tab("识别"):
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
    with gr.Tab("身份库"):
        refresh = gr.Button("刷新身份库")
        identities_table = gr.Dataframe(
            headers=["身份", "姓名", "注册图数量"],
            datatype=["str", "str", "number"],
        )
        refresh.click(load_identities, outputs=identities_table)


if __name__ == "__main__":
    demo.launch()
