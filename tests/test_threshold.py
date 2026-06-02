from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts.analyze_threshold import build_parser, main
from src.eval.threshold import SimilarityDistributions, collect_similarity_distributions, suggest_threshold


def unit(values: list[float]) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def create_registered_fixture(root: Path) -> Path:
    write_image(root / "p01" / "r1.png", (255, 0, 0))
    write_image(root / "p01" / "r2.png", (255, 0, 0))
    write_image(root / "p02" / "r1.png", (0, 255, 0))
    write_image(root / "p02" / "r2.png", (0, 255, 0))
    return root


def test_collect_similarity_distributions_and_suggest_threshold() -> None:
    embeddings_by_identity = {
        "p01": [unit([1, 0, 0]), unit([1, 0, 0])],
        "p02": [unit([0, 1, 0]), unit([0, 1, 0])],
    }

    distributions = collect_similarity_distributions(embeddings_by_identity)
    threshold = suggest_threshold(distributions)

    assert distributions == SimilarityDistributions(
        same_identity=[1.0, 1.0],
        different_identity=[0.0, 0.0, 0.0, 0.0],
    )
    assert 0.0 < threshold < 1.0


def test_suggest_threshold_raises_when_distribution_is_missing() -> None:
    with pytest.raises(ValueError):
        suggest_threshold(SimilarityDistributions(same_identity=[], different_identity=[0.1]))


def test_analyze_threshold_returns_friendly_error_when_registered_root_missing(tmp_path, capsys) -> None:
    histogram_path = tmp_path / "reports" / "threshold_hist.png"

    exit_code = main(
        [
            "--registered-root",
            str(tmp_path / "missing"),
            "--histogram-path",
            str(histogram_path),
            "--model-name",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "不存在" in captured.out
    assert not histogram_path.exists()


def test_analyze_threshold_parser_defaults_to_dataset_registered() -> None:
    args = build_parser().parse_args([])

    assert args.registered_root == "dataset/registered"


def test_analyze_threshold_script_writes_histogram_with_fake_model(tmp_path) -> None:
    registered_root = create_registered_fixture(tmp_path / "registered")
    histogram_path = tmp_path / "reports" / "threshold_hist.png"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_threshold.py",
            "--registered-root",
            str(registered_root),
            "--histogram-path",
            str(histogram_path),
            "--model-name",
            "fake",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "suggested threshold" in result.stdout
    assert histogram_path.exists()
