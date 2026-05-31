from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CelebaSample:
    path: Path
    identity_id: str


@dataclass(frozen=True)
class CelebaDataset:
    register_root: Path
    test_root: Path
    identity_ids: list[str]
    test_samples: list[CelebaSample]


def load_celeba_dataset(root: str | Path) -> CelebaDataset:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"CelebA 数据目录不存在: {root}")

    register_root = root / "register"
    test_root = root / "test"
    if not register_root.exists():
        raise FileNotFoundError(f"CelebA register 目录不存在: {register_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"CelebA test 目录不存在: {test_root}")

    identity_ids = sorted(path.name for path in register_root.iterdir() if path.is_dir())
    test_samples: list[CelebaSample] = []
    for identity_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        for image_path in sorted(identity_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            test_samples.append(CelebaSample(path=image_path, identity_id=identity_dir.name))

    return CelebaDataset(
        register_root=register_root,
        test_root=test_root,
        identity_ids=identity_ids,
        test_samples=test_samples,
    )
