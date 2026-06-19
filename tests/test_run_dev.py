import importlib.util
from pathlib import Path

from src.face_model import model_config


def _load_run_dev_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_dev.py"
    spec = importlib.util.spec_from_file_location("test_run_dev_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_popen(run_dev, monkeypatch):
    """Stub subprocess.Popen, return the list it records (command, env) into."""
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
    monkeypatch.setattr(run_dev, "_missing_dependencies", lambda: [])
    return popen_calls


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
    # run_dev now only spawns the backend; the frontend is served by FastAPI itself.
    assert len(popen_calls) == 1
    assert popen_calls[0][0] == ["python", "-m", "uvicorn", "src.backend.api:app", "--port", "8000"]
    assert popen_calls[0][1]["FACEPASS_MODEL_PATH"] == "C:/models/buffalo_l"


def test_run_dev_uses_config_when_persisted_path_is_valid(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    popen_calls = _capture_popen(run_dev, monkeypatch)

    monkeypatch.setattr(model_config, "load_persisted_path", lambda: Path("C:/saved/buffalo_l"))
    monkeypatch.setattr(model_config, "validate_model_path", lambda path: Path(path))
    # A valid saved path must not trigger any prompt.
    monkeypatch.setattr(run_dev, "_prompt", lambda text: (_ for _ in ()).throw(AssertionError("should not prompt")))

    result = run_dev.main([])

    assert result == 0
    assert len(popen_calls) == 1
    # Backend reads config.toml itself; no env override is injected.
    assert "FACEPASS_MODEL_PATH" not in popen_calls[0][1]


def test_run_dev_prompts_and_persists_when_user_accepts(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    popen_calls = _capture_popen(run_dev, monkeypatch)
    persisted: list[Path] = []

    monkeypatch.setattr(model_config, "load_persisted_path", lambda: None)
    monkeypatch.setattr(model_config, "validate_model_path", lambda path: Path(path))
    monkeypatch.setattr(model_config, "persist_path", persisted.append)
    monkeypatch.setattr(run_dev, "_prompt", lambda text: "D:/models/buffalo_l")
    monkeypatch.setattr(run_dev, "_ask_yes_no", lambda *args, **kwargs: True)

    result = run_dev.main([])

    assert result == 0
    assert persisted == [Path("D:/models/buffalo_l")]
    # Saved to config, so the backend reads it from there — no env override.
    assert "FACEPASS_MODEL_PATH" not in popen_calls[0][1]


def test_run_dev_uses_session_path_without_persisting_when_user_declines(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    popen_calls = _capture_popen(run_dev, monkeypatch)
    persisted: list[Path] = []

    monkeypatch.setattr(model_config, "load_persisted_path", lambda: None)
    monkeypatch.setattr(model_config, "validate_model_path", lambda path: Path(path))
    monkeypatch.setattr(model_config, "persist_path", persisted.append)
    monkeypatch.setattr(run_dev, "_prompt", lambda text: "D:/models/buffalo_l")
    monkeypatch.setattr(run_dev, "_ask_yes_no", lambda *args, **kwargs: False)

    result = run_dev.main([])

    assert result == 0
    assert persisted == []
    # Used for this session only, passed via env, not written to config.
    assert popen_calls[0][1]["FACEPASS_MODEL_PATH"] == str(Path("D:/models/buffalo_l"))


def test_run_dev_aborts_when_no_model_path_provided(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    popen_calls = _capture_popen(run_dev, monkeypatch)

    monkeypatch.setattr(model_config, "load_persisted_path", lambda: None)
    monkeypatch.setattr(run_dev, "_prompt", lambda text: "")  # empty input => abort

    result = run_dev.main([])

    assert result == 1
    assert popen_calls == []


def test_run_dev_reprompts_when_persisted_path_invalid(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    popen_calls = _capture_popen(run_dev, monkeypatch)
    from src.common.errors import ModelNotFoundError

    monkeypatch.setattr(model_config, "load_persisted_path", lambda: Path("C:/gone/buffalo_l"))

    def fake_validate(path):
        if str(path).replace("\\", "/") == "C:/gone/buffalo_l":
            raise ModelNotFoundError("missing")
        return Path(path)

    monkeypatch.setattr(model_config, "validate_model_path", fake_validate)
    monkeypatch.setattr(model_config, "persist_path", lambda path: None)
    monkeypatch.setattr(run_dev, "_prompt", lambda text: "D:/models/buffalo_l")
    monkeypatch.setattr(run_dev, "_ask_yes_no", lambda *args, **kwargs: True)

    result = run_dev.main([])

    assert result == 0
    assert len(popen_calls) == 1
