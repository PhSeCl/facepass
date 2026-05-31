import numpy as np
import pytest
from PIL import Image

from src.common.errors import EmptyGalleryError
from src.backend.gallery import Gallery


def unit(values: list[float]) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_match_uses_best_similarity_across_identity_embeddings() -> None:
    gallery = Gallery()
    gallery.register("p01", [unit([1, 0, 0]), unit([0, 1, 0])])
    gallery.register("p02", [unit([0, 0, 1])])

    identity_id, similarity = gallery.match(unit([0.05, 0.95, 0]))

    assert identity_id == "p01"
    assert similarity > 0.99


def test_match_returns_none_for_empty_gallery() -> None:
    gallery = Gallery()

    assert gallery.match(unit([1, 0, 0])) is None


def test_save_and_load_preserve_registered_embeddings(tmp_path) -> None:
    path = tmp_path / "gallery.pkl"
    gallery = Gallery()
    gallery.register("p01", [unit([1, 0, 0])])

    gallery.save(path)
    loaded = Gallery.load(path)

    assert loaded.identities() == [{"identity_id": "p01", "count": 1}]
    identity_id, similarity = loaded.match(unit([1, 0, 0]))
    assert identity_id == "p01"
    assert similarity == 1.0


class FakeBuildModel:
    def encode_aligned(self, face_image: np.ndarray) -> np.ndarray:
        return unit([1, 0, 0])

    def detect_and_encode(self, image: np.ndarray):
        return []


def test_build_from_dir_skips_bad_images_and_registers_valid_images(tmp_path) -> None:
    root = tmp_path / "registered"
    identity_dir = root / "p01"
    identity_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(identity_dir / "valid.png")
    (identity_dir / "broken.jpg").write_bytes(b"not an image")

    gallery = Gallery()
    gallery.build_from_dir(str(root), FakeBuildModel())

    assert gallery.identities() == [{"identity_id": "p01", "count": 1}]


def test_build_from_dir_raises_when_no_valid_registration_images(tmp_path) -> None:
    root = tmp_path / "registered"
    identity_dir = root / "p01"
    identity_dir.mkdir(parents=True)
    (identity_dir / "broken.jpg").write_bytes(b"not an image")

    gallery = Gallery()

    with pytest.raises(EmptyGalleryError):
        gallery.build_from_dir(str(root), FakeBuildModel())
