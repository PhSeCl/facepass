import csv
import base64
import re
import tempfile
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
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
from .dir_picker import DirectoryPickerUnavailable, pick_directory_via_dialog
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
from .schemas import (
    BatchRegisterResponse,
    ConfusionPairModel,
    DatasetEvalResponse,
    DatasetInspectResponse,
    DetectionIssueModel,
    ErrorResponse,
    EvalMetricsModel,
    EvalPlotsModel,
    IdentitiesResponse,
    IdentityDetail,
    IdentitySummary,
    PickDirectoryResponse,
    RecognitionResultModel,
    RegisterResponse,
)


logger = get_logger(__name__)
MODEL_PATH_ENV_VAR = "FACEPASS_MODEL_PATH"


def _validate_dataset_dir(dataset_dir: str) -> Path:
    # Normalize first so "..", relative segments, and symlinks collapse, then
    # validate the real target. This is a local single-user tool whose UI
    # invites pointing at any local dataset directory, so the guard is path
    # normalization plus an existence check — not a "blocklist .." substring
    # rule (which both false-rejects legitimate paths and fails to actually
    # confine reads to a root).
    resolved = Path(dataset_dir).resolve()
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail={"message": f"数据集路径不存在或不是目录: {resolved}"})
    return resolved


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(8192)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail={"message": "上传文件过大"})
        chunks.append(chunk)
    return b"".join(chunks)


_recognizer: Recognizer | None = None
_gallery: Gallery = Gallery()
_id2name: dict[str, str] = {}


_IDENTITY_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_identities(path: Path | None = None) -> dict[str, str]:
    path = path or settings.identities_csv
    if not path.exists():
        logger.warning("身份映射文件不存在: %s", path)
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["identity_id"]: row["name"] for row in csv.DictReader(handle)}


def _read_identity_rows(path: Path | None = None) -> dict[str, dict[str, str]]:
    path = path or settings.identities_csv
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: dict[str, dict[str, str]] = {}
        for row in csv.DictReader(handle):
            identity_id = (row.get("identity_id") or "").strip()
            if not identity_id:
                continue
            rows[identity_id] = {
                "name": row.get("name") or "",
                "domain": row.get("domain") or "",
            }
    return rows


def _write_identity_name(identity_id: str, name: str) -> None:
    """Upsert a single identity name while preserving any existing domain column."""
    rows = _read_identity_rows()
    existing = rows.get(identity_id, {})
    rows[identity_id] = {"name": name, "domain": existing.get("domain", "")}
    settings.identities_csv.parent.mkdir(parents=True, exist_ok=True)
    with settings.identities_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["identity_id", "name", "domain"])
        for iid, data in sorted(rows.items()):
            writer.writerow([iid, data["name"], data.get("domain", "")])
    _id2name[identity_id] = name


def _validate_identity_id(identity_id: str) -> str:
    identity_id = identity_id.strip()
    if not identity_id or not _IDENTITY_ID_PATTERN.fullmatch(identity_id):
        raise HTTPException(
            status_code=400,
            detail={"message": "identity_id 非法：仅允许字母、数字、下划线和连字符，且不能为空"},
        )
    return identity_id


def _resolve_extension(filename: str | None) -> str:
    ext = Path(filename or "image.jpg").suffix.lower()
    return ext if ext in _ALLOWED_IMAGE_EXTS else ".jpg"


def _next_registered_index(identity_dir: Path, identity_id: str) -> int:
    pattern = re.compile(rf"{re.escape(identity_id)}_r(\d+)$")
    max_index = 0
    for path in identity_dir.glob(f"{identity_id}_r*"):
        match = pattern.match(path.stem)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _rebuild_gallery(model) -> None:
    global _gallery, _recognizer
    _gallery = Gallery()
    _gallery.build_from_dir(str(settings.registered_dir), model)
    _gallery.save(settings.gallery_path)
    _recognizer = Recognizer(model, _gallery, settings.threshold, _id2name)
    logger.info("底库已更新")


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
            cached_gallery = Gallery.load(settings.gallery_path)
            if getattr(cached_gallery, "requires_rebuild", False):
                _gallery = Gallery()
                _gallery.build_from_dir(str(settings.registered_dir), model)
                _gallery.save(settings.gallery_path)
            else:
                _gallery = cached_gallery
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "static" / "index.html"


@app.get("/")
def root():
    if _FRONTEND_HTML.exists():
        return HTMLResponse(content=_FRONTEND_HTML.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h2>FacePass API</h2>")


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


def _build_eval_response(result) -> DatasetEvalResponse:
    """Render plots and assemble the eval response. Runs synchronously (matplotlib);
    call via run_in_threadpool so it does not block the event loop."""
    metrics = result.report.metrics
    plots = _render_external_eval_plots(result.dataset, result.report)
    return DatasetEvalResponse(
        gallery_source=result.gallery_source,
        metrics=EvalMetricsModel(
            strict_top1_accuracy=metrics.strict_top1_accuracy,
            matched_top1_accuracy=metrics.matched_top1_accuracy,
            detection_recall=metrics.detection_recall,
            detection_precision=metrics.detection_precision,
            unknown_detected_accuracy=metrics.unknown_detected_accuracy,
            predicted_unknown_precision=metrics.predicted_unknown_precision,
        ),
        plots=EvalPlotsModel(
            confusion_matrix=plots["confusion_matrix"],
            detection_metrics=plots["detection_metrics"],
            accuracy_metrics=plots["accuracy_metrics"],
        ),
        confusion_pairs=[
            ConfusionPairModel(
                true_identity_id=true_identity_id,
                predicted_identity_id=predicted_identity_id,
            )
            for true_identity_id, predicted_identity_id in result.confusion_pairs
        ],
        missed_detections=[
            DetectionIssueModel(image_name=item.image_name, bbox=list(item.bbox))
            for item in result.missed_detections
        ],
        false_positives=[
            DetectionIssueModel(image_name=item.image_name, bbox=list(item.bbox))
            for item in result.false_positives
        ],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/logo")
def logo():
    logo_path = Path(__file__).resolve().parents[2] / "design" / "facepass.png"
    if not logo_path.exists():
        raise HTTPException(status_code=404, detail={"message": "Logo not found"})
    return FileResponse(logo_path, media_type="image/png")


@app.get("/favicon.ico")
def favicon():
    # Browsers auto-request /favicon.ico; without a route it 404s and logs a
    # spurious WARNING on every page load. Reuse the logo, or answer 204.
    logo_path = Path(__file__).resolve().parents[2] / "design" / "facepass.png"
    if logo_path.exists():
        return FileResponse(logo_path, media_type="image/png")
    return Response(status_code=204)


@app.get("/identity/{identity_id}/image")
def identity_image(identity_id: str):
    identity_dir = settings.registered_dir / identity_id
    if not identity_dir.exists() or not identity_dir.is_dir():
        raise HTTPException(status_code=404, detail={"message": f"身份不存在: {identity_id}"})
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for child in sorted(identity_dir.iterdir()):
        if child.suffix.lower() in image_extensions:
            return FileResponse(child, media_type=f"image/{child.suffix.lstrip('.').replace('jpg', 'jpeg')}")
    raise HTTPException(status_code=404, detail={"message": f"身份 {identity_id} 没有注册图片"})


@app.get("/identity/{identity_id}/image/{filename}")
def identity_image_file(identity_id: str, filename: str):
    identity_dir = settings.registered_dir / identity_id
    if not identity_dir.exists() or not identity_dir.is_dir():
        raise HTTPException(status_code=404, detail={"message": f"身份不存在: {identity_id}"})
    # Guard against path traversal: the filename must resolve to a direct child of
    # identity_dir (a bare name with no separators / "..").
    if Path(filename).name != filename:
        raise HTTPException(status_code=404, detail={"message": f"图片不存在: {filename}"})
    image_path = identity_dir / filename
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail={"message": f"图片不存在: {filename}"})
    suffix = image_path.suffix.lstrip(".").replace("jpg", "jpeg")
    return FileResponse(image_path, media_type=f"image/{suffix}")


@app.get("/identity/{identity_id}/detail", response_model=IdentityDetail)
def identity_detail(identity_id: str) -> IdentityDetail:
    identity_dir = settings.registered_dir / identity_id
    if not identity_dir.exists() or not identity_dir.is_dir():
        raise HTTPException(status_code=404, detail={"message": f"身份不存在: {identity_id}"})
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        child.name
        for child in identity_dir.iterdir()
        if child.is_file() and child.suffix.lower() in image_extensions
    )
    if not images:
        raise HTTPException(status_code=404, detail={"message": f"身份 {identity_id} 没有注册图片"})
    stats = next((item for item in _gallery.identities() if str(item["identity_id"]) == identity_id), {})
    return IdentityDetail(
        identity_id=identity_id,
        name=_id2name.get(identity_id),
        valid_image_count=int(stats.get("valid_image_count", len(images))),
        prototype_count=int(stats.get("prototype_count", 1)),
        images=images,
    )


@app.get("/identities", response_model=IdentitiesResponse)
def identities() -> IdentitiesResponse:
    summaries = [
        IdentitySummary(
            identity_id=item["identity_id"],
            name=_id2name.get(str(item["identity_id"])),
            count=int(item["count"]),
            prototype_count=int(item["prototype_count"]),
            valid_image_count=int(item["valid_image_count"]),
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
    content = await _read_upload_limited(file, settings.max_upload_bytes)
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


@app.post(
    "/register",
    response_model=RegisterResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def register(
    file: UploadFile = File(...),
    identity_id: str = Form(...),
    name: str = Form(""),
) -> RegisterResponse:
    identity_id = _validate_identity_id(identity_id)
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={"message": "上传图片过大"})
    try:
        image = safe_load_image(content)
    except InvalidImageError as exc:
        logger.warning("拒绝无效注册图片 %s: %s", file.filename, exc)
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    recognizer = get_recognizer()
    if recognizer is None:
        raise HTTPException(status_code=503, detail={"message": "识别器未初始化"})

    faces = recognizer.model.detect_and_encode(image)
    if not faces:
        raise HTTPException(status_code=400, detail={"message": "未检测到人脸"})

    identity_dir = settings.registered_dir / identity_id
    identity_dir.mkdir(parents=True, exist_ok=True)
    index = _next_registered_index(identity_dir, identity_id)
    image_path = identity_dir / f"{identity_id}_r{index:02d}{_resolve_extension(file.filename)}"
    image_path.write_bytes(content)

    if name:
        _write_identity_name(identity_id, name)

    _rebuild_gallery(recognizer.model)

    return RegisterResponse(identity_id=identity_id, name=_id2name.get(identity_id, ""))


@app.post(
    "/register/batch",
    response_model=BatchRegisterResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def register_batch(
    files: list[UploadFile] = File(...),
    identity_id: str = Form(...),
    name: str = Form(""),
) -> BatchRegisterResponse:
    identity_id = _validate_identity_id(identity_id)
    recognizer = get_recognizer()
    if recognizer is None:
        raise HTTPException(status_code=503, detail={"message": "识别器未初始化"})

    identity_dir = settings.registered_dir / identity_id
    identity_dir.mkdir(parents=True, exist_ok=True)
    next_index = _next_registered_index(identity_dir, identity_id)

    saved = 0
    for file in files:
        content = await file.read()
        if len(content) > settings.max_upload_bytes:
            continue
        try:
            image = safe_load_image(content)
        except InvalidImageError:
            continue
        if not recognizer.model.detect_and_encode(image):
            continue
        image_path = identity_dir / f"{identity_id}_r{next_index:02d}{_resolve_extension(file.filename)}"
        image_path.write_bytes(content)
        next_index += 1
        saved += 1

    if saved == 0:
        raise HTTPException(status_code=400, detail={"message": "没有成功录入任何图片"})

    if name:
        _write_identity_name(identity_id, name)

    _rebuild_gallery(recognizer.model)

    return BatchRegisterResponse(
        identity_id=identity_id,
        name=_id2name.get(identity_id, ""),
        saved=saved,
    )


@app.post(
    "/pick-directory",
    response_model=PickDirectoryResponse,
    responses={400: {"model": ErrorResponse}},
)
async def pick_directory() -> PickDirectoryResponse:
    # Show the OS-native folder picker on the machine running the backend (same
    # desktop as the browser for this local tool) and return the chosen absolute
    # path so the frontend can fill it into the dataset-directory field.
    try:
        path = await run_in_threadpool(pick_directory_via_dialog)
    except DirectoryPickerUnavailable as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    return PickDirectoryResponse(path=path)


@app.post(
    "/dataset-eval/inspect",
    response_model=DatasetInspectResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def inspect_dataset_archive(
    file: UploadFile | None = File(None),
    dataset_dir: str | None = Form(None),
) -> DatasetInspectResponse:
    if file is not None and dataset_dir:
        raise HTTPException(status_code=400, detail={"message": "请只提供 zip 或数据集目录中的一种"})
    if file is None and not dataset_dir:
        raise HTTPException(status_code=400, detail={"message": "请先上传 test.zip 或选择数据集文件夹"})

    if dataset_dir:
        validated_dir = _validate_dataset_dir(dataset_dir)
        try:
            has_registered = inspect_external_dataset_directory(str(validated_dir))
        except (DatasetArchiveError, DatasetLayoutError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
        return DatasetInspectResponse(has_registered=has_registered)

    assert file is not None
    content = await _read_upload_limited(file, settings.max_upload_bytes)

    archive_path = _write_temp_upload(content, Path(file.filename or "dataset.zip").suffix or ".zip")
    try:
        has_registered = inspect_external_dataset_archive(archive_path)
    except (DatasetArchiveError, DatasetLayoutError) as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    finally:
        archive_path.unlink(missing_ok=True)
    return DatasetInspectResponse(has_registered=has_registered)


@app.post(
    "/dataset-eval/run",
    response_model=DatasetEvalResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def evaluate_dataset_archive(
    gallery_choice: str = Form("local"),
    dataset_dir: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> DatasetEvalResponse:
    recognizer = get_recognizer()
    if recognizer is None:
        raise HTTPException(status_code=503, detail={"message": "识别器未初始化"})

    if file is not None and dataset_dir:
        raise HTTPException(status_code=400, detail={"message": "请只提供 zip 或数据集目录中的一种"})
    if file is None and not dataset_dir:
        raise HTTPException(status_code=400, detail={"message": "请先上传 test.zip 或选择数据集文件夹"})

    try:
        if dataset_dir:
            validated_dir = _validate_dataset_dir(dataset_dir)
            result = await run_in_threadpool(
                run_external_eval_from_directory,
                str(validated_dir),
                gallery_choice,
                model=recognizer.model,
                threshold=settings.threshold,
                local_registered_root=settings.registered_dir,
                local_gallery=_gallery,
            )
        else:
            assert file is not None
            content = await _read_upload_limited(file, settings.max_upload_bytes)

            archive_path = _write_temp_upload(content, Path(file.filename or "dataset.zip").suffix or ".zip")
            try:
                result = await run_in_threadpool(
                    run_external_eval,
                    archive_path,
                    gallery_choice,
                    model=recognizer.model,
                    threshold=settings.threshold,
                    local_registered_root=settings.registered_dir,
                    local_gallery=_gallery,
                )
            finally:
                archive_path.unlink(missing_ok=True)

        return await run_in_threadpool(_build_eval_response, result)
    except (DatasetArchiveError, DatasetLayoutError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
