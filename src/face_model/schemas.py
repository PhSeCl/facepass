from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    # Optional metadata; consumers should not assume this field is used.
    det_score: float
    # Optional metadata; consumers should not assume this field is used.
    landmarks: np.ndarray | None = None
