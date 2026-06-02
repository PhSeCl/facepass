import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from scripts.eval_celeba import build_parser, main


def write_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def create_celeba_fixture(root: Path) -> Path:
    data_dir = root / "celeba_100_identities_3reg_3test"
    register_dir = data_dir / "register"
    test_dir = data_dir / "test"

    write_image(
        register_dir / "identity_00070" / "107551.jpg",
        np.full((8, 8, 3), fill_value=(255, 0, 0), dtype=np.uint8),
    )
    write_image(
        register_dir / "identity_00212" / "000001.jpg",
        np.full((8, 8, 3), fill_value=(0, 255, 0), dtype=np.uint8),
    )

    write_image(
        test_dir / "identity_00070" / "151880.jpg",
        np.full((8, 8, 3), fill_value=(255, 0, 0), dtype=np.uint8),
    )
    write_image(
        test_dir / "identity_00070" / "151881.jpg",
        np.full((8, 8, 3), fill_value=(0, 255, 0), dtype=np.uint8),
    )
    write_image(
        test_dir / "identity_00212" / "000002.jpg",
        np.full((8, 8, 3), fill_value=(0, 255, 0), dtype=np.uint8),
    )
    return data_dir


def test_eval_celeba_returns_friendly_error_when_data_dir_missing(
    tmp_path: Path,
    capsys,
) -> None:
    report_path = tmp_path / "reports" / "celeba_eval.json"

    exit_code = main(
        [
            "--data-dir",
            str(tmp_path / "missing"),
            "--report-path",
            str(report_path),
            "--model-name",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "CelebA 评测失败" in captured.out
    assert not report_path.exists()


def test_eval_celeba_parser_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.data_dir == "celeba_100_identities_3reg_3test"
    assert args.report_path == "reports/celeba_eval.json"
    assert args.top1_plot_path == "reports/celeba_top1_accuracy.png"
    assert args.per_class_plot_path == "reports/celeba_per_class_accuracy.png"
    assert args.sample_limit == 5


def test_eval_celeba_script_runs_with_fake_model_and_writes_report(tmp_path: Path) -> None:
    data_dir = create_celeba_fixture(tmp_path)
    report_path = tmp_path / "reports" / "celeba_eval.json"
    top1_plot_path = tmp_path / "reports" / "celeba_top1_accuracy.png"
    per_class_plot_path = tmp_path / "reports" / "celeba_per_class_accuracy.png"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_celeba.py",
            "--data-dir",
            str(data_dir),
            "--report-path",
            str(report_path),
            "--top1-plot-path",
            str(top1_plot_path),
            "--per-class-plot-path",
            str(per_class_plot_path),
            "--model-name",
            "fake",
            "--sample-limit",
            "2",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert "top-1 accuracy" in result.stdout
    assert report["top1_accuracy"] == 2 / 3
    assert report["total"] == 3
    assert report["correct"] == 2
    assert len(report["success_samples"]) == 2
    assert len(report["failure_samples"]) == 1
    assert report["failure_samples"][0]["true_identity_id"] == "identity_00070"
    assert report["failure_samples"][0]["predicted_identity_id"] == "identity_00212"
    assert top1_plot_path.exists()
    assert per_class_plot_path.exists()
    assert "top-1 plot written to" in result.stdout
    assert "per-class plot written to" in result.stdout
