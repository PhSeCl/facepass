from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_name: str = "insightface"
    threshold: float = 0.30
    gallery_path: Path = ROOT_DIR / "models" / "gallery.pkl"
    registered_dir: Path = ROOT_DIR / "data" / "registered"
    identities_csv: Path = ROOT_DIR / "dataset" / "identities.csv"
    max_upload_bytes: int = 10 * 1024 * 1024


settings = Settings()

# The default threshold is only for running the minimal closed loop. The final
# threshold should be chosen from registered-set similarity distributions, never
# tuned on the held-out test set.
