from dataclasses import dataclass

import numpy as np

from src.face_model.base import FaceModel

from .gallery import Gallery
from .schemas import RecognitionResult


@dataclass(frozen=True)
class RecognitionPreview:
    bbox: tuple[int, int, int, int]
    best_identity_id: str
    similarity: float
    is_unknown: bool


class Recognizer:
    def __init__(
        self,
        model: FaceModel,
        gallery: Gallery,
        threshold: float,
        id2name: dict[str, str],
    ) -> None:
        self.model = model
        self.gallery = gallery
        self.threshold = threshold
        self.id2name = id2name

    def preview_image(self, image: np.ndarray) -> list[RecognitionPreview]:
        previews: list[RecognitionPreview] = []
        for face in self.model.detect_and_encode(image):
            match = self.gallery.match(face.embedding)
            if match is None:
                best_identity_id = ""
                similarity = 0.0
                is_unknown = True
            else:
                best_identity_id, similarity = match
                is_unknown = similarity < self.threshold
            previews.append(
                RecognitionPreview(
                    bbox=face.bbox,
                    best_identity_id=best_identity_id,
                    similarity=float(similarity),
                    is_unknown=is_unknown,
                )
            )
        return previews

    def recognize_image(self, image: np.ndarray) -> list[RecognitionResult]:
        results: list[RecognitionResult] = []
        for preview in self.preview_image(image):
            identity_id = preview.best_identity_id
            similarity = preview.similarity
            is_unknown = preview.is_unknown
            output_id = "unknown" if is_unknown else identity_id
            results.append(
                RecognitionResult(
                    bbox=preview.bbox,
                    identity_id=output_id,
                    name=None if is_unknown else self.id2name.get(identity_id),
                    similarity=float(similarity),
                    is_unknown=is_unknown,
                )
            )
        return results
