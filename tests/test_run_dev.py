import importlib.util
from pathlib import Path


def _load_run_dev_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_dev.py"
    spec = importlib.util.spec_from_file_location("test_run_dev_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_dev_passes_model_path_to_backend_env(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    popen_calls: list[tuple[list[str], dict[str, str] | None]] = []

    class FakeProcess:
        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            return None

    def fake_popen(command, env=None):
        popen_calls.append((command, env))
        return FakeProcess()

    monkeypatch.setattr(run_dev.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_dev.sys, "executable", "python")

    result = run_dev.main(["--model-path", "C:/models/buffalo_l"])

    assert result == 0
    assert popen_calls[0][0] == ["python", "-m", "uvicorn", "src.backend.api:app", "--port", "8000"]
    assert popen_calls[0][1]["FACEPASS_MODEL_PATH"] == "C:/models/buffalo_l"
    assert popen_calls[1][0] == ["python", "src/frontend/app.py"]
