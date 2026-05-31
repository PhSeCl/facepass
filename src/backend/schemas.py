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


class IdentitiesResponse(BaseModel):
    identities: list[IdentitySummary]
