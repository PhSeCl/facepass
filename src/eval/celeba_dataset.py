from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.errors import InvalidImageError
from src.common.images import safe_load_image
from src.common.logging import get_logger


logger = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CelebATestSample:
    face_image: np.ndarray
    identity_id: str
    source_path: Path


@dataclass(frozen=True)
class CelebADataset:
    root: Path
    register_dir: Path
    test_dir: Path
    registered_images: dict[str, list[np.ndarray]]
    samples: list[CelebATestSample]


def _iter_identity_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def _load_images(identity_dir: Path) -> list[tuple[Path, np.ndarray]]:
    images: list[tuple[Path, np.ndarray]] = []
    for image_path in sorted(identity_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            images.append((image_path, safe_load_image(image_path)))
        except (InvalidImageError, OSError) as exc:
            logger.warning("跳过无效 CelebA 图片 %s: %s", image_path, exc)
    return images


def load_celeba_dataset(root: str | Path) -> CelebADataset:
    root = Path(root)
    register_dir = root / "register"
    test_dir = root / "test"

    if not root.exists():
        raise FileNotFoundError(f"CelebA 数据目录不存在: {root}")
    if not register_dir.exists():
        raise FileNotFoundError(f"CelebA register 目录不存在: {register_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"CelebA test 目录不存在: {test_dir}")

    register_identity_dirs = _iter_identity_dirs(register_dir)
    test_identity_dirs = _iter_identity_dirs(test_dir)
    register_identities = {path.name for path in register_identity_dirs}
    test_identities = {path.name for path in test_identity_dirs}
    if register_identities != test_identities:
        missing_in_test = sorted(register_identities - test_identities)
        missing_in_register = sorted(test_identities - register_identities)
        details: list[str] = []
        if missing_in_test:
            details.append(f"test 缺少: {', '.join(missing_in_test)}")
        if missing_in_register:
            details.append(f"register 缺少: {', '.join(missing_in_register)}")
        raise ValueError(f"CelebA register/test 身份目录不一致: {'; '.join(details)}")

    registered_images: dict[str, list[np.ndarray]] = {}
    samples: list[CelebATestSample] = []
    for identity_id in sorted(register_identities):
        register_images = _load_images(register_dir / identity_id)
        test_images = _load_images(test_dir / identity_id)
        if not register_images or not test_images:
            logger.warning("跳过没有有效图片的 CelebA 身份 %s", identity_id)
            continue
        registered_images[identity_id] = [image for _, image in register_images]
        for image_path, image in test_images:
            samples.append(
                CelebATestSample(
                    face_image=image,
                    identity_id=identity_id,
                    source_path=image_path,
                )
            )

    if not registered_images:
        raise ValueError(f"CelebA register 中没有可用身份: {register_dir}")
    if not samples:
        raise ValueError(f"CelebA test 中没有可用样本: {test_dir}")

    return CelebADataset(
        root=root,
        register_dir=register_dir,
        test_dir=test_dir,
        registered_images=registered_images,
        samples=samples,
    )
