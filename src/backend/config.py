from dataclasses import dataclass, field
from pathlib import Path

from src.face_model import model_config


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD = 0.30


def _load_threshold() -> float:
    data = model_config.load_config_data()
    recognition_section = data.get("recognition")
    if not isinstance(recognition_section, dict):
        return DEFAULT_THRESHOLD

    threshold = recognition_section.get("threshold")
    if isinstance(threshold, (int, float)):
        return float(threshold)
    return DEFAULT_THRESHOLD


@dataclass(frozen=True)
class Settings:
    model_name: str = "insightface"
    threshold: float = field(default_factory=_load_threshold)
    gallery_path: Path = ROOT_DIR / "models" / "gallery.npz"
    registered_dir: Path = ROOT_DIR / "dataset" / "registered"
    identities_csv: Path = ROOT_DIR / "dataset" / "identities.csv"
    max_upload_bytes: int = 10 * 1024 * 1024


settings = Settings()

# TODO: replace the default threshold with a value chosen from real evaluation
# reports once registration/test data collection is complete.
