import json
import subprocess
import sys
from pathlib import Path


def test_check_runtime_script_runs_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/check_runtime.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "runtime" in payload
    assert "preferred_providers" in payload["runtime"]
