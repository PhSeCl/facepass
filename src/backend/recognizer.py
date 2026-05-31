import numpy as np

from src.face_model.base import FaceModel

from .gallery import Gallery
from .schemas import RecognitionResult


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

    def recognize_image(self, image: np.ndarray) -> list[RecognitionResult]:
        results: list[RecognitionResult] = []
        for face in self.model.detect_and_encode(image):
            match = self.gallery.match(face.embedding)
            if match is None:
                identity_id = ""
                similarity = 0.0
                is_unknown = True
            else:
                identity_id, similarity = match
                is_unknown = similarity < self.threshold
            output_id = "unknown" if is_unknown else identity_id
            results.append(
                RecognitionResult(
                    bbox=face.bbox,
                    identity_id=output_id,
                    name=None if is_unknown else self.id2name.get(identity_id),
                    similarity=float(similarity),
                    is_unknown=is_unknown,
                )
            )
        return results
