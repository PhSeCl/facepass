import csv
import base64
import tempfile
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.common.errors import (
    EmptyGalleryError,
    FatalStartupError,
    InvalidImageError,
    ModelIncompleteError,
    ModelLoadError,
    ModelNotFoundError,
    ModelPathMissingError,
)
from src.common.images import safe_load_image
from src.common.logging import get_logger
from src.eval.end2end_reporting import (
    plot_accuracy_metrics,
    plot_confusion_matrix,
    plot_detection_metrics,
)
from src.face_model import create_model

from .config import settings
from .dataset_import import (
    DatasetArchiveError,
    DatasetLayoutError,
    inspect_external_dataset_archive,
    inspect_external_dataset_directory,
    run_external_eval,
    run_external_eval_from_directory,
)
from .gallery import Gallery
from .recognizer import Recognizer
from .schemas import ErrorResponse, IdentitiesResponse, IdentitySummary, RecognitionResultModel


logger = get_logger(__name__)
MODEL_PATH_ENV_VAR = "FACEPASS_MODEL_PATH"
_recognizer: Recognizer | None = None
_gallery: Gallery = Gallery()
_id2name: dict[str, str] = {}


def load_identities(path: Path | None = None) -> dict[str, str]:
    path = path or settings.identities_csv
    if not path.exists():
        logger.warning("身份映射文件不存在: %s", path)
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["identity_id"]: row["name"] for row in csv.DictReader(handle)}


def _startup_failure_message(exc: Exception) -> str:
    if isinstance(exc, ModelPathMissingError):
        return "后端启动失败: 缺少模型路径。请使用 --model-path 指定 buffalo_l 目录，或在 config.toml 中设置 [model].path。"
    if isinstance(exc, ModelNotFoundError):
        return f"后端启动失败: 模型路径无效。{exc}。请检查 --model-path 或 config.toml 是否指向现有的 buffalo_l 目录。"
    if isinstance(exc, ModelIncompleteError):
        return f"后端启动失败: 模型目录不完整。{exc}。请检查 buffalo_l 目录中的 ONNX 文件是否齐全且非空。"
    if isinstance(exc, ModelLoadError):
        return f"后端启动失败: 模型文件可能损坏或不兼容。{exc}。请检查路径下的模型文件，必要时重新获取模型。"
    return f"后端启动失败: {exc}"


def startup(
    fail_fast: bool = True,
    cli_model_path: str | Path | None = None,
    gui_model_path: str | Path | None = None,
) -> None:
    global _recognizer, _gallery, _id2name
    _id2name = load_identities(settings.identities_csv)
    if cli_model_path is None:
        cli_model_path = os.environ.get(MODEL_PATH_ENV_VAR)
    try:
        model_kwargs = {}
        if settings.model_name.lower() == "insightface":
            model_kwargs = {
                "model_path": cli_model_path,
                "gui_model_path": gui_model_path,
            }
        model = create_model(settings.model_name, **model_kwargs)
        if settings.gallery_path.exists():
            _gallery = Gallery.load(settings.gallery_path)
        else:
            _gallery = Gallery()
            _gallery.build_from_dir(str(settings.registered_dir), model)
            _gallery.save(settings.gallery_path)
        _recognizer = Recognizer(model, _gallery, settings.threshold, _id2name)
        logger.info("后端初始化完成")
    except (ModelPathMissingError, ModelNotFoundError, ModelIncompleteError, ModelLoadError) as exc:
        message = _startup_failure_message(exc)
        logger.error("%s", message)
        if fail_fast:
            sys.exit(1)
        raise FatalStartupError(message) from exc
    except (EmptyGalleryError, OSError, RuntimeError, ImportError) as exc:
        message = _startup_failure_message(exc)
        logger.error("%s。请检查模型权重、注册集目录和配置。", message)
        if fail_fast:
            sys.exit(1)
        raise FatalStartupError(message) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup(fail_fast=True)
    yield


app = FastAPI(title="FacePass API", lifespan=lifespan)


@app.exception_handler(ModelLoadError)
async def model_load_exception_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
    logger.error("模型运行失败 %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"message": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("未预期请求错误 %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"message": "服务器内部错误，请查看后端日志"})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "message" in detail:
        content = {"message": detail["message"]}
    else:
        content = {"message": str(detail)}
    if exc.status_code >= 500:
        logger.error("HTTP 错误 %s %s: %s", request.method, request.url.path, content["message"])
    else:
        logger.warning("HTTP 错误 %s %s: %s", request.method, request.url.path, content["message"])
    return JSONResponse(status_code=exc.status_code, content=content)


def get_recognizer() -> Recognizer | None:
    return _recognizer


def _write_temp_upload(content: bytes, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="facepass-upload-", suffix=suffix, delete=False)
    try:
        handle.write(content)
    finally:
        handle.close()
    return Path(handle.name)


def _png_to_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _render_external_eval_plots(dataset, report) -> dict[str, str]:
    temp_dir = Path(tempfile.mkdtemp(prefix="facepass-eval-plots-"))
    try:
        confusion_path = temp_dir / "confusion.png"
        detection_path = temp_dir / "detection.png"
        accuracy_path = temp_dir / "accuracy.png"
        plot_confusion_matrix(dataset, report, confusion_path)
        plot_detection_metrics(report, detection_path)
        plot_accuracy_metrics(report, accuracy_path)
        return {
            "confusion_matrix": _png_to_data_url(confusion_path),
            "detection_metrics": _png_to_data_url(detection_path),
            "accuracy_metrics": _png_to_data_url(accuracy_path),
        }
    finally:
        for child in temp_dir.glob("*"):
            child.unlink(missing_ok=True)
        temp_dir.rmdir()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/identities", response_model=IdentitiesResponse)
def identities() -> IdentitiesResponse:
    summaries = [
        IdentitySummary(
            identity_id=item["identity_id"],
            name=_id2name.get(str(item["identity_id"])),
            count=int(item["count"]),
        )
        for item in _gallery.identities()
    ]
    return IdentitiesResponse(identities=summaries)


@app.post(
    "/recognize",
    response_model=list[RecognitionResultModel],
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def recognize(file: UploadFile = File(...)) -> list[RecognitionResultModel]:
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={"message": "上传图片过大"})
    try:
        image = safe_load_image(content)
    except InvalidImageError as exc:
        logger.warning("拒绝无效上传图片 %s: %s", file.filename, exc)
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    recognizer = get_recognizer()
    if recognizer is None:
        logger.warning("识别器未初始化，返回空识别结果")
        return []
    results = recognizer.recognize_image(image)
    return [RecognitionResultModel(**result.__dict__) for result in results]


@app.post("/dataset-eval/inspect", responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}})
async def inspect_dataset_archive(
    file: UploadFile | None = File(None),
    dataset_dir: str | None = Form(None),
) -> dict[str, bool]:
    if file is not None and dataset_dir:
        raise HTTPException(status_code=400, detail={"message": "请只提供 zip 或数据集目录中的一种"})
    if file is None and not dataset_dir:
        raise HTTPException(status_code=400, detail={"message": "请先上传 test.zip 或选择数据集文件夹"})

    if dataset_dir:
        try:
            return {"has_registered": inspect_external_dataset_directory(dataset_dir)}
        except (DatasetArchiveError, DatasetLayoutError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    assert file is not None
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={"message": "上传压缩包过大"})

    archive_path = _write_temp_upload(content, Path(file.filename or "dataset.zip").suffix or ".zip")
    try:
        has_registered = inspect_external_dataset_archive(archive_path)
    except (DatasetArchiveError, DatasetLayoutError) as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    finally:
        archive_path.unlink(missing_ok=True)
    return {"has_registered": has_registered}


@app.post("/dataset-eval/run", responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}})
async def evaluate_dataset_archive(
    gallery_choice: str = Form("local"),
    dataset_dir: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> dict[str, object]:
    recognizer = get_recognizer()
    if recognizer is None:
        raise HTTPException(status_code=503, detail={"message": "识别器未初始化"})

    if file is not None and dataset_dir:
        raise HTTPException(status_code=400, detail={"message": "请只提供 zip 或数据集目录中的一种"})
    if file is None and not dataset_dir:
        raise HTTPException(status_code=400, detail={"message": "请先上传 test.zip 或选择数据集文件夹"})

    try:
        if dataset_dir:
            result = run_external_eval_from_directory(
                dataset_dir,
                gallery_choice,
                model=recognizer.model,
                threshold=settings.threshold,
                local_registered_root=settings.registered_dir,
                local_gallery=_gallery,
            )
        else:
            assert file is not None
            content = await file.read()
            if len(content) > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail={"message": "上传压缩包过大"})

            archive_path = _write_temp_upload(content, Path(file.filename or "dataset.zip").suffix or ".zip")
            try:
                result = run_external_eval(
                    archive_path,
                    gallery_choice,
                    model=recognizer.model,
                    threshold=settings.threshold,
                    local_registered_root=settings.registered_dir,
                    local_gallery=_gallery,
                )
            finally:
                archive_path.unlink(missing_ok=True)

        metrics = result.report.metrics
        return {
            "gallery_source": result.gallery_source,
            "metrics": {
                "strict_top1_accuracy": metrics.strict_top1_accuracy,
                "matched_top1_accuracy": metrics.matched_top1_accuracy,
                "detection_recall": metrics.detection_recall,
                "detection_precision": metrics.detection_precision,
                "unknown_detected_accuracy": metrics.unknown_detected_accuracy,
                "predicted_unknown_precision": metrics.predicted_unknown_precision,
            },
            "plots": _render_external_eval_plots(result.dataset, result.report),
            "confusion_pairs": [
                {
                    "true_identity_id": true_identity_id,
                    "predicted_identity_id": predicted_identity_id,
                }
                for true_identity_id, predicted_identity_id in result.confusion_pairs
            ],
            "missed_detections": [
                {"image_name": item.image_name, "bbox": list(item.bbox)}
                for item in result.missed_detections
            ],
            "false_positives": [
                {"image_name": item.image_name, "bbox": list(item.bbox)}
                for item in result.false_positives
            ],
        }
    except (DatasetArchiveError, DatasetLayoutError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
