from pathlib import Path

import pytest

from src.face_model.insightface_model import InsightFaceModel
from src.face_model.model_config import BUFFALO_L_REQUIRED_FILES, load_persisted_path, resolve_model_path


def _configured_model_path() -> Path | None:
    model_path = load_persisted_path()
    if model_path is None:
        return None
    if not model_path.is_dir():
        return None
    for filename in BUFFALO_L_REQUIRED_FILES:
        candidate = model_path / filename
        if not candidate.exists() or candidate.stat().st_size <= 0:
            return None
    return model_path


def _sample_face_image_path() -> Path | None:
    candidates = [
        Path(".venv/Lib/site-packages/insightface/data/images/t1.jpg"),
        Path(".venv/Lib/site-packages/insightface/data/images/Tom_Hanks_54745.png"),
        Path(".venv/Lib/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg"),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


REAL_MODEL_PATH = _configured_model_path()
SAMPLE_FACE_IMAGE = _sample_face_image_path()


@pytest.mark.skipif(REAL_MODEL_PATH is None, reason="real buffalo_l model files are not configured locally")
@pytest.mark.skipif(SAMPLE_FACE_IMAGE is None, reason="no local face sample image is available for the smoke test")
def test_real_buffalo_l_smoke_loads_and_detects_face() -> None:
    import cv2

    resolved_path = resolve_model_path()
    assert resolved_path == REAL_MODEL_PATH

    model = InsightFaceModel()
    image = cv2.imread(str(SAMPLE_FACE_IMAGE))

    assert image is not None

    faces = model.detect_and_encode(image)

    assert len(faces) >= 1
    assert faces[0].bbox[2] > 0
    assert faces[0].bbox[3] > 0
    assert faces[0].embedding.shape == (512,)
