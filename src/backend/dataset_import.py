from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from src.common.errors import FacePassError
from src.eval.end2end_dataset import GroupedSelfDataset, load_grouped_self_dataset
from src.eval.end2end_evaluator import EndToEndEvalReport, evaluate_end2end
from src.face_model.base import FaceModel


ANNOTATION_FILENAMES = (
    "annotation.json",
    "annotation.jsonl",
    "annotations.json",
    "annotations.jsonl",
)


class DatasetArchiveError(FacePassError):
    """Raised when an uploaded dataset archive cannot be trusted or extracted."""


class DatasetLayoutError(FacePassError):
    """Raised when an extracted archive does not match the expected dataset layout."""


@dataclass(frozen=True)
class ExtractedDatasetArchive:
    extracted_root: Path
    dataset_root: Path
    images_dir: Path
    annotation_path: Path
    annotation_format: str
    registered_dir: Path | None


@dataclass(frozen=True)
class DetectionIssue:
    image_name: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class ExternalEvalResult:
    gallery_source: str
    dataset: GroupedSelfDataset
    report: EndToEndEvalReport
    confusion_pairs: list[tuple[str, str]]
    missed_detections: list[DetectionIssue]
    false_positives: list[DetectionIssue]


def _find_7z_executable() -> str | None:
    return shutil.which("7z") or shutil.which("7za")


def _validate_member_name(name: str) -> None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if normalized.startswith("/") or normalized.startswith("\\"):
        raise DatasetArchiveError(f"压缩包包含不安全路径: {name}")
    if path.is_absolute():
        raise DatasetArchiveError(f"压缩包包含不安全路径: {name}")
    if any(part == ".." for part in path.parts):
        raise DatasetArchiveError(f"压缩包包含不安全路径: {name}")
    if path.parts and path.parts[0].endswith(":"):
        raise DatasetArchiveError(f"压缩包包含不安全路径: {name}")


def _read_validated_members(archive_path: Path) -> list[zipfile.ZipInfo]:
    try:
        with zipfile.ZipFile(archive_path, "r") as handle:
            members = handle.infolist()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise DatasetArchiveError("压缩包无法解压,可能已损坏") from exc

    for member in members:
        _validate_member_name(member.filename)
    return members


def _extract_with_7z(zip_path: Path, destination: Path, executable: str) -> None:
    result = subprocess.run(
        [executable, "x", str(zip_path), f"-o{destination}", "-y"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "压缩包无法解压,可能已损坏"
        raise DatasetArchiveError(f"压缩包无法解压,可能已损坏: {message}")


def _extract_with_zipfile(zip_path: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path, "r") as handle:
            for member in handle.infolist():
                _validate_member_name(member.filename)
                relative = PurePosixPath(member.filename.replace("\\", "/"))
                if not relative.parts:
                    continue
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member, "r") as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise DatasetArchiveError("压缩包无法解压,可能已损坏") from exc


def _locate_dataset_root(extracted_root: Path) -> ExtractedDatasetArchive:
    candidates: list[ExtractedDatasetArchive] = []
    search_roots = [extracted_root, *sorted((path for path in extracted_root.rglob("*") if path.is_dir()), key=lambda p: len(p.parts))]
    for candidate_root in search_roots:
        images_dir = candidate_root / "images"
        if not images_dir.is_dir():
            continue
        annotation_paths = [candidate_root / filename for filename in ANNOTATION_FILENAMES if (candidate_root / filename).is_file()]
        if not annotation_paths:
            continue
        if len(annotation_paths) > 1:
            names = ", ".join(path.name for path in annotation_paths)
            raise DatasetLayoutError(f"解压成功但找到多个标注文件: {names}")
        annotation_path = annotation_paths[0]
        candidates.append(
            ExtractedDatasetArchive(
                extracted_root=extracted_root,
                dataset_root=candidate_root,
                images_dir=images_dir,
                annotation_path=annotation_path,
                annotation_format=annotation_path.suffix.lstrip("."),
                registered_dir=(candidate_root / "registered") if (candidate_root / "registered").is_dir() else None,
            )
        )

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        locations = ", ".join(str(candidate.dataset_root.relative_to(extracted_root)) or "." for candidate in candidates)
        raise DatasetLayoutError(f"解压成功但检测到多个候选数据集目录: {locations}")

    visible = ", ".join(sorted(path.name for path in extracted_root.iterdir())) or "(empty)"
    raise DatasetLayoutError(f"解压成功但未找到 images/ 或标注文件; 顶层包含: {visible}")


def extract_dataset_archive(archive_path: str | Path) -> ExtractedDatasetArchive:
    archive_path = Path(archive_path)
    _read_validated_members(archive_path)

    extracted_root = Path(tempfile.mkdtemp(prefix="facepass-dataset-"))
    try:
        executable = _find_7z_executable()
        if executable:
            _extract_with_7z(archive_path, extracted_root, executable)
        else:
            _extract_with_zipfile(archive_path, extracted_root)
        return _locate_dataset_root(extracted_root)
    except Exception:
        shutil.rmtree(extracted_root, ignore_errors=True)
        raise


def inspect_external_dataset_archive(archive_path: str | Path) -> bool:
    extracted_root: Path | None = None
    try:
        extracted = extract_dataset_archive(archive_path)
        extracted_root = extracted.extracted_root
        return extracted.registered_dir is not None
    finally:
        if extracted_root is not None:
            shutil.rmtree(extracted_root, ignore_errors=True)


def _resolve_registered_root(
    extracted: ExtractedDatasetArchive,
    gallery_choice: str,
    local_registered_root: Path,
) -> tuple[str, Path]:
    normalized_choice = gallery_choice.lower()
    if normalized_choice not in {"local", "archive"}:
        raise ValueError(f"unsupported gallery choice: {gallery_choice}")
    if normalized_choice == "archive" and extracted.registered_dir is not None:
        return "archive", extracted.registered_dir
    return "local", local_registered_root


def _with_registered_root(dataset: GroupedSelfDataset, registered_root: Path) -> GroupedSelfDataset:
    return GroupedSelfDataset(
        registered_root=registered_root,
        test_root=dataset.test_root,
        images=dataset.images,
    )


def _collect_detection_issues(report: EndToEndEvalReport, dataset: GroupedSelfDataset) -> tuple[list[DetectionIssue], list[DetectionIssue]]:
    missed_detections: list[DetectionIssue] = []
    false_positives: list[DetectionIssue] = []

    for image_result, sample in zip(report.image_results, dataset.images):
        for ground_truth_index in image_result.unmatched_ground_truth_indices:
            missed_detections.append(
                DetectionIssue(
                    image_name=sample.image_path.name,
                    bbox=image_result.ground_truths[ground_truth_index].bbox,
                )
            )
        for prediction_index in image_result.unmatched_prediction_indices:
            false_positives.append(
                DetectionIssue(
                    image_name=sample.image_path.name,
                    bbox=image_result.predictions[prediction_index].bbox,
                )
            )
    return missed_detections, false_positives


def run_external_eval(
    zip_path: str | Path,
    gallery_choice: str,
    *,
    model: FaceModel,
    threshold: float,
    local_registered_root: str | Path,
) -> ExternalEvalResult:
    extracted_root: Path | None = None
    try:
        extracted = extract_dataset_archive(zip_path)
        extracted_root = extracted.extracted_root
        gallery_source, registered_root = _resolve_registered_root(
            extracted,
            gallery_choice,
            Path(local_registered_root),
        )
        dataset = load_grouped_self_dataset(
            annotations_path=extracted.annotation_path,
            test_root=extracted.dataset_root,
            registered_root=registered_root,
        )
        dataset = _with_registered_root(dataset, registered_root)
        report = evaluate_end2end(dataset=dataset, model=model, threshold=threshold)
        missed_detections, false_positives = _collect_detection_issues(report, dataset)
        return ExternalEvalResult(
            gallery_source=gallery_source,
            dataset=dataset,
            report=report,
            confusion_pairs=report.metrics.confusion_pairs,
            missed_detections=missed_detections,
            false_positives=false_positives,
        )
    finally:
        if extracted_root is not None:
            shutil.rmtree(extracted_root, ignore_errors=True)
