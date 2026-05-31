from abc import ABC, abstractmethod

import numpy as np

from .schemas import DetectedFace


class FaceModel(ABC):
    @abstractmethod
    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        """Input a BGR image array and return all detected faces.

        Returned bbox values must be integer absolute-pixel (x, y, w, h), and
        embeddings must be L2-normalized.
        """

    @abstractmethod
    def encode_aligned(self, face_image: np.ndarray) -> np.ndarray:
        """Encode an already-cropped BGR face image without detection."""
