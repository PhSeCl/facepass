from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class RecognitionResult:
    bbox: tuple[int, int, int, int]
    identity_id: str
    name: str | None
    similarity: float
    is_unknown: bool


class RecognitionResultModel(BaseModel):
    bbox: tuple[int, int, int, int]
    identity_id: str
    name: str | None
    similarity: float
    is_unknown: bool


class ErrorResponse(BaseModel):
    message: str


class IdentitySummary(BaseModel):
    identity_id: str
    name: str | None = None
    count: int
    prototype_count: int
    valid_image_count: int


class IdentitiesResponse(BaseModel):
    identities: list[IdentitySummary]


class RegisterResponse(BaseModel):
    identity_id: str
    name: str


class BatchRegisterResponse(BaseModel):
    identity_id: str
    name: str
    saved: int


class DatasetInspectResponse(BaseModel):
    has_registered: bool


class EvalMetricsModel(BaseModel):
    strict_top1_accuracy: float
    matched_top1_accuracy: float
    detection_recall: float
    detection_precision: float
    unknown_detected_accuracy: float
    predicted_unknown_precision: float


class EvalPlotsModel(BaseModel):
    confusion_matrix: str
    detection_metrics: str
    accuracy_metrics: str


class ConfusionPairModel(BaseModel):
    true_identity_id: str
    predicted_identity_id: str


class DetectionIssueModel(BaseModel):
    image_name: str
    bbox: list[int]


class IdentityDetail(BaseModel):
    identity_id: str
    name: str | None = None
    valid_image_count: int
    prototype_count: int
    images: list[str]


class PickDirectoryResponse(BaseModel):
    path: str


class DatasetEvalResponse(BaseModel):
    gallery_source: str
    metrics: EvalMetricsModel
    plots: EvalPlotsModel
    confusion_pairs: list[ConfusionPairModel]
    missed_detections: list[DetectionIssueModel]
    false_positives: list[DetectionIssueModel]
