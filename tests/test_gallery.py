import numpy as np
import pytest
from PIL import Image

from src.common.errors import EmptyGalleryError
from src.backend.gallery import Gallery
from src.face_model.schemas import DetectedFace


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
        height, width = image.shape[:2]
        return [
            DetectedFace(
                bbox=(0, 0, int(width), int(height)),
                embedding=unit([1, 0, 0]),
                det_score=1.0,
                landmarks=None,
            )
        ]


class FakeDetectBuildModel:
    def encode_aligned(self, face_image: np.ndarray) -> np.ndarray:
        raise AssertionError("build_from_dir should use detect_and_encode for registration images")

    def detect_and_encode(self, image: np.ndarray):
        signature = tuple(int(value) for value in image[0, 0, :3])
        if signature == (0, 0, 255):  # red image after BGR conversion
            return [
                DetectedFace(
                    bbox=(0, 0, 10, 10),
                    embedding=unit([1, 0, 0]),
                    det_score=1.0,
                    landmarks=None,
                )
            ]
        if signature == (0, 255, 0):  # green image
            return [
                DetectedFace(
                    bbox=(0, 0, 10, 10),
                    embedding=unit([0, 1, 0]),
                    det_score=1.0,
                    landmarks=None,
                )
            ]
        if signature == (255, 0, 0):  # blue image
            return [
                DetectedFace(
                    bbox=(0, 0, 4, 4),
                    embedding=unit([1, 0, 0]),
                    det_score=0.9,
                    landmarks=None,
                ),
                DetectedFace(
                    bbox=(0, 0, 8, 8),
                    embedding=unit([0, 0, 1]),
                    det_score=0.8,
                    landmarks=None,
                ),
            ]
        if signature == (255, 255, 255):  # white image
            return []
        raise AssertionError(f"unexpected test image signature: {signature}")


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


def test_build_from_dir_detects_faces_selects_largest_and_averages_per_identity(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "registered"
    identity_one = root / "p01"
    identity_two = root / "p02"
    identity_one.mkdir(parents=True)
    identity_two.mkdir(parents=True)

    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(identity_one / "red.png")
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(identity_one / "green.png")
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(identity_two / "multi-face.png")
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(identity_two / "no-face.png")

    gallery = Gallery()
    with caplog.at_level("WARNING"):
        gallery.build_from_dir(str(root), FakeDetectBuildModel())

    assert gallery.identities() == [
        {"identity_id": "p01", "count": 1},
        {"identity_id": "p02", "count": 1},
    ]
    identity_id, similarity = gallery.match(unit([1, 1, 0]))
    assert identity_id == "p01"
    assert similarity > 0.99
    identity_id, similarity = gallery.match(unit([0, 0, 1]))
    assert identity_id == "p02"
    assert similarity > 0.99
    assert "检测到多张人脸" in caplog.text
    assert "未检测到人脸" in caplog.text


class FakeCroppedBuildModel:
    def __init__(self) -> None:
        self.detect_calls = 0
        self.encode_calls = 0

    def encode_aligned(self, face_image: np.ndarray) -> np.ndarray:
        self.encode_calls += 1
        signature = tuple(int(value) for value in face_image[0, 0, :3])
        dominant_channel = int(np.argmax(signature))
        if dominant_channel == 2:
            return unit([1, 0, 0])
        if dominant_channel == 1:
            return unit([0, 1, 0])
        raise AssertionError(f"unexpected test image signature: {signature}")

    def detect_and_encode(self, image: np.ndarray):
        self.detect_calls += 1
        raise AssertionError("cropped-face gallery build must not call detect_and_encode")


def test_build_from_cropped_dir_uses_encode_aligned_and_averages_identity_embeddings(
    tmp_path,
) -> None:
    root = tmp_path / "celeba_register"
    identity_one = root / "identity_00070"
    identity_two = root / "identity_00212"
    identity_one.mkdir(parents=True)
    identity_two.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(identity_one / "107551.jpg")
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(identity_one / "130995.jpg")
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(identity_two / "000001.jpg")

    model = FakeCroppedBuildModel()
    gallery = Gallery()

    gallery.build_from_cropped_dir(str(root), model)

    assert gallery.identities() == [
        {"identity_id": "identity_00070", "count": 1},
        {"identity_id": "identity_00212", "count": 1},
    ]
    identity_id, similarity = gallery.match(unit([1, 0, 0]))
    assert identity_id == "identity_00070"
    assert similarity == pytest.approx(1.0)
    identity_id, similarity = gallery.match(unit([0, 1, 0]))
    assert identity_id == "identity_00212"
    assert similarity == pytest.approx(1.0)
    assert model.encode_calls == 3
    assert model.detect_calls == 0
