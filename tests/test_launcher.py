import importlib.util
from pathlib import Path

import pytest


def _load_launcher():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "launcher.py"
    spec = importlib.util.spec_from_file_location("facepass_launcher", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher()


def test_required_modules_excludes_onnxruntime() -> None:
    modules = launcher.required_modules()
    assert "onnxruntime" not in modules
    # Sanity: still contains other backend deps.
    assert "fastapi" in modules
    assert modules["cv2"] == "opencv-python"


def test_read_runtime_parses_table() -> None:
    text = (
        '[model]\npath = "x"\n\n'
        '[runtime]\ndevice = "gpu"\nlauncher = "venv"\n'
    )
    assert launcher.read_runtime(text) == {"device": "gpu", "launcher": "venv"}


def test_read_runtime_absent_returns_none() -> None:
    assert launcher.read_runtime('[model]\npath = "x"\n') is None


def test_strip_runtime_preserves_other_sections() -> None:
    text = (
        '[model]\npath = "x"\n\n'
        '[runtime]\ndevice = "cpu"\nlauncher = "uv"\n\n'
        '[recognition]\nthreshold = 0.3\n'
    )
    stripped = launcher.strip_runtime(text)
    assert "[runtime]" not in stripped
    assert "[model]" in stripped
    assert 'path = "x"' in stripped
    assert "[recognition]" in stripped
    assert "threshold = 0.3" in stripped


def test_save_runtime_roundtrip_and_preserves(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[model]\npath = "F:/m"\n\n[recognition]\nthreshold = 0.3\n', encoding="utf-8")
    monkeypatch.setattr(launcher, "CONFIG_PATH", config)

    launcher.save_runtime("gpu", "venv")

    text = config.read_text(encoding="utf-8")
    assert launcher.read_runtime(text) == {"device": "gpu", "launcher": "venv"}
    # Original sections survive.
    assert 'path = "F:/m"' in text
    assert "threshold = 0.3" in text

    # Saving again replaces, not appends, the runtime table.
    launcher.save_runtime("cpu", "uv")
    text2 = config.read_text(encoding="utf-8")
    assert text2.count("[runtime]") == 1
    assert launcher.read_runtime(text2) == {"device": "cpu", "launcher": "uv"}


def test_save_runtime_creates_file_when_absent(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    monkeypatch.setattr(launcher, "CONFIG_PATH", config)
    launcher.save_runtime("cpu", "global")
    assert launcher.read_runtime(config.read_text(encoding="utf-8")) == {
        "device": "cpu",
        "launcher": "global",
    }


def test_resolve_launch_command_tokens(monkeypatch) -> None:
    assert launcher.resolve_launch_command("uv") == ["uv", "run", "python"]
    assert launcher.resolve_launch_command("venv") == [str(launcher.VENV_PY)]
    assert launcher.resolve_launch_command("venv-gpu") == [str(launcher.GPU_VENV_PY)]
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "C:/py/python.exe")
    monkeypatch.setattr(launcher, "_python_works", lambda interp: True)
    assert launcher.resolve_launch_command("global") == ["C:/py/python.exe"]
    assert launcher.resolve_launch_command("bogus") is None


def test_find_global_python_rejects_broken_interpreter(monkeypatch) -> None:
    # `python` resolves but does not actually run (e.g. the Store stub).
    monkeypatch.setattr(launcher.shutil, "which", lambda name: f"C:/{name}.exe")
    monkeypatch.setattr(launcher, "_python_works", lambda interp: False)
    assert launcher.find_global_python() is None


def test_find_global_python_falls_back_to_py_launcher(monkeypatch) -> None:
    monkeypatch.setattr(launcher.shutil, "which", lambda name: f"C:/{name}.exe")
    # python.exe is a non-working stub; only the `py` launcher runs.
    monkeypatch.setattr(launcher, "_python_works", lambda interp: interp == "py")
    assert launcher.find_global_python() == "py"


def test_runtime_is_valid_accepts_dedicated_gpu_venv(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_check_interpreter", lambda token: "GPUPY")
    monkeypatch.setattr(launcher, "missing_packages", lambda interp: [])
    monkeypatch.setattr(launcher, "has_onnxruntime", lambda interp: True)
    monkeypatch.setattr(launcher, "cuda_available", lambda interp: True)
    assert launcher.runtime_is_valid({"launcher": "venv-gpu", "device": "gpu"}, has_uv=False) is True


def test_runtime_is_valid_happy_path(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_check_interpreter", lambda token: "PY")
    monkeypatch.setattr(launcher, "missing_packages", lambda interp: [])
    monkeypatch.setattr(launcher, "has_onnxruntime", lambda interp: True)
    monkeypatch.setattr(launcher, "cuda_available", lambda interp: True)

    assert launcher.runtime_is_valid({"launcher": "venv", "device": "cpu"}, has_uv=False) is True
    assert launcher.runtime_is_valid({"launcher": "venv", "device": "gpu"}, has_uv=False) is True


def test_runtime_is_valid_rejects_missing_deps(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_check_interpreter", lambda token: "PY")
    monkeypatch.setattr(launcher, "missing_packages", lambda interp: ["fastapi"])
    monkeypatch.setattr(launcher, "has_onnxruntime", lambda interp: True)
    assert launcher.runtime_is_valid({"launcher": "venv", "device": "cpu"}, has_uv=False) is False


def test_runtime_is_valid_rejects_gpu_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_check_interpreter", lambda token: "PY")
    monkeypatch.setattr(launcher, "missing_packages", lambda interp: [])
    monkeypatch.setattr(launcher, "has_onnxruntime", lambda interp: True)
    monkeypatch.setattr(launcher, "cuda_available", lambda interp: False)
    assert launcher.runtime_is_valid({"launcher": "venv", "device": "gpu"}, has_uv=False) is False


def test_runtime_is_valid_rejects_uv_without_uv(monkeypatch) -> None:
    assert launcher.runtime_is_valid({"launcher": "uv", "device": "cpu"}, has_uv=False) is False


def test_runtime_is_valid_rejects_bad_tokens() -> None:
    assert launcher.runtime_is_valid({"launcher": "nope", "device": "cpu"}, has_uv=True) is False
    assert launcher.runtime_is_valid({"launcher": "venv", "device": "tpu"}, has_uv=True) is False


def test_report_unexpected_error_points_to_issue_tracker(capsys, monkeypatch) -> None:
    # Decline the export prompt so the test does not write a file.
    monkeypatch.setattr(launcher, "ask_yes_no", lambda *a, **k: False)
    try:
        raise RuntimeError("boom-xyz")
    except RuntimeError:
        launcher._report_unexpected_error()
    out = capsys.readouterr().out
    assert "启动失败" in out
    assert launcher.ISSUE_URL in out  # tells the user where to report
    assert "boom-xyz" in out  # includes the actual traceback


def test_write_crash_report_creates_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    path = launcher._write_crash_report("Traceback: boom-detail")
    assert path is not None
    assert path.parent == tmp_path
    assert path.suffix == ".log"
    text = path.read_text(encoding="utf-8")
    assert "boom-detail" in text
    assert launcher.ISSUE_URL in text  # report header carries the feedback link
