import argparse
import hashlib
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval.celeba_dataset import IMAGE_EXTENSIONS, _iter_identity_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether CelebA register/test splits contain identical image files."
    )
    parser.add_argument(
        "--data-dir",
        default="celeba_100_identities_3reg_3test",
        help="CelebA root directory containing register/ and test/.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_identity_dir(identity_dir: Path) -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = {}
    for image_path in sorted(identity_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        file_hash = _sha256(image_path)
        hashes.setdefault(file_hash, []).append(image_path.name)
    return hashes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.data_dir)
    register_dir = root / "register"
    test_dir = root / "test"

    if not root.exists():
        print(f"CelebA 数据目录不存在: {root}")
        return 1
    if not register_dir.exists():
        print(f"CelebA register 目录不存在: {register_dir}")
        return 1
    if not test_dir.exists():
        print(f"CelebA test 目录不存在: {test_dir}")
        return 1

    register_identity_dirs = _iter_identity_dirs(register_dir)
    test_identity_dirs = _iter_identity_dirs(test_dir)
    register_identities = {path.name for path in register_identity_dirs}
    test_identities = {path.name for path in test_identity_dirs}
    if register_identities != test_identities:
        print("CelebA register/test 身份目录不一致，无法继续做泄漏检查。")
        return 1

    overlapping_identities: list[str] = []
    total_overlap_hashes = 0
    for identity_id in sorted(register_identities):
        register_hashes = _hash_identity_dir(register_dir / identity_id)
        test_hashes = _hash_identity_dir(test_dir / identity_id)
        overlap_hashes = sorted(register_hashes.keys() & test_hashes.keys())
        if overlap_hashes:
            overlapping_identities.append(identity_id)
            total_overlap_hashes += len(overlap_hashes)
            print(f"[overlap] {identity_id}")
            for file_hash in overlap_hashes:
                register_files = ", ".join(register_hashes[file_hash])
                test_files = ", ".join(test_hashes[file_hash])
                print(f"  hash={file_hash}")
                print(f"  register: {register_files}")
                print(f"  test: {test_files}")

    if overlapping_identities:
        print(
            f"发现 {len(overlapping_identities)} 个身份存在跨 register/test 的相同文件哈希，"
            f"共 {total_overlap_hashes} 个重叠哈希。"
        )
        return 2

    print("未发现任何跨 register/test 的相同文件哈希。")
    print(f"已检查 {len(register_identities)} 个身份。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
