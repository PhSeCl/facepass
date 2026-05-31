import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8), mode="RGB").save(path)


def test_visualize_embeddings_script_writes_tsne_plot(tmp_path) -> None:
    images_root = tmp_path / "images"
    write_image(images_root / "p01_a.png", (255, 0, 0))
    write_image(images_root / "p01_b.png", (255, 0, 0))
    write_image(images_root / "p02_a.png", (0, 255, 0))
    write_image(images_root / "p02_b.png", (0, 0, 255))

    report_path = tmp_path / "reports" / "eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "samples": [
            {
                "path": str(images_root / "p01_a.png"),
                "true_identity_id": "p01",
                "predicted_identity_id": "p01",
                "similarity": 1.0,
            },
            {
                "path": str(images_root / "p01_b.png"),
                "true_identity_id": "p01",
                "predicted_identity_id": "p01",
                "similarity": 1.0,
            },
            {
                "path": str(images_root / "p02_a.png"),
                "true_identity_id": "p02",
                "predicted_identity_id": "p02",
                "similarity": 1.0,
            },
            {
                "path": str(images_root / "p02_b.png"),
                "true_identity_id": "p02",
                "predicted_identity_id": "p01",
                "similarity": 0.0,
            },
            {
                "path": str(images_root / "p02_b.png"),
                "bbox": [0, 0, 8, 8],
                "true_identity_id": "unknown",
                "predicted_identity_id": "unknown",
                "similarity": 0.0,
            },
        ]
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    output_path = tmp_path / "reports" / "tsne.png"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_embeddings.py",
            "--report-path",
            str(report_path),
            "--output-path",
            str(output_path),
            "--model-name",
            "fake",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "t-SNE figure written to" in result.stdout
    assert output_path.exists()
