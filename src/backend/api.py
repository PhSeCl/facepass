import csv
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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
from src.face_model import create_model

from .config import settings
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
