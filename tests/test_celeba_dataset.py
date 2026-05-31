import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from src.eval.celeba_dataset import CelebaSample, load_celeba_dataset
from scripts.eval_celeba import main


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def create_fixture(root: Path) -> Path:
    write_image(root / "register" / "p01" / "r1.png", (255, 0, 0))
    write_image(root / "register" / "p02" / "r1.png", (0, 255, 0))
    write_image(root / "test" / "p01" / "t1.png", (255, 0, 0))
    write_image(root / "test" / "p02" / "t1.png", (0, 255, 0))
    return root


def test_load_celeba_dataset_reads_register_and_test_layout(tmp_path) -> None:
    root = create_fixture(tmp_path / "celeba")

    dataset = load_celeba_dataset(root)

    assert dataset.register_root == root / "register"
    assert dataset.identity_ids == ["p01", "p02"]
    assert dataset.test_root == root / "test"
    assert dataset.test_samples == [
        CelebaSample(path=root / "test" / "p01" / "t1.png", identity_id="p01"),
        CelebaSample(path=root / "test" / "p02" / "t1.png", identity_id="p02"),
    ]


def test_eval_celeba_returns_friendly_error_when_dataset_missing(tmp_path, capsys) -> None:
    report_path = tmp_path / "reports" / "celeba_eval.json"

    exit_code = main(
        [
            "--dataset-root",
            str(tmp_path / "missing"),
            "--report-path",
            str(report_path),
            "--model-name",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "不存在" in captured.out
    assert not report_path.exists()


def test_eval_celeba_writes_report_with_fake_model(tmp_path, capsys) -> None:
    root = create_fixture(tmp_path / "celeba")
    report_path = tmp_path / "reports" / "celeba_eval.json"

    exit_code = main(
        [
            "--dataset-root",
            str(root),
            "--report-path",
            str(report_path),
            "--model-name",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "top-1 accuracy" in captured.out
    assert report["top1_accuracy"] == 1.0
    assert report["total"] == 2
    assert report["correct"] == 2
    assert len(report["samples"]) == 2
    assert len(report["success_samples"]) == 2
    assert report["failure_samples"] == []


def test_eval_celeba_script_runs_as_cli_entrypoint(tmp_path) -> None:
    root = create_fixture(tmp_path / "celeba")
    report_path = tmp_path / "reports" / "celeba_eval_cli.json"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_celeba.py",
            "--dataset-root",
            str(root),
            "--report-path",
            str(report_path),
            "--model-name",
            "fake",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "top-1 accuracy" in result.stdout
    assert report_path.exists()
