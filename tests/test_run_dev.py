import importlib.util
import subprocess
from pathlib import Path


def _load_run_dev_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_dev.py"
    spec = importlib.util.spec_from_file_location("test_run_dev_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    """Configurable stand-in for subprocess.Popen.

    ``poll_results`` is consumed one value per call (the last value sticks),
    so a process can report "still running" (None) and then "exited".
    """

    def __init__(self, poll_results=None, wait_raises_timeout=False):
        self._poll_results = list(poll_results) if poll_results else [None]
        self._wait_raises_timeout = wait_raises_timeout
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        if len(self._poll_results) > 1:
            return self._poll_results.pop(0)
        return self._poll_results[0]

    def terminate(self):
        self.terminated = True
        self._poll_results = [-15]

    def kill(self):
        self.killed = True
        self._poll_results = [-9]

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._wait_raises_timeout and not self.killed:
            raise subprocess.TimeoutExpired(cmd="run_dev", timeout=timeout)
        return self._poll_results[-1]


def test_run_dev_passes_model_path_to_backend_env(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    popen_calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_popen(command, env=None):
        popen_calls.append((command, env))
        # Report the process as already exited so _supervise returns at once.
        return FakeProcess(poll_results=[0])

    monkeypatch.setattr(run_dev.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_dev.sys, "executable", "python")

    result = run_dev.main(["--model-path", "C:/models/buffalo_l"])

    assert result == 0
    assert popen_calls[0][0] == ["python", "-m", "uvicorn", "src.backend.api:app", "--port", "8000"]
    assert popen_calls[0][1]["FACEPASS_MODEL_PATH"] == "C:/models/buffalo_l"
    assert popen_calls[1][0] == ["python", "src/frontend/app.py"]


def test_run_dev_stops_frontend_when_backend_exits(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    backend = FakeProcess(poll_results=[1])  # backend crashed
    frontend = FakeProcess(poll_results=[None])  # frontend still running
    processes = iter([backend, frontend])

    monkeypatch.setattr(run_dev.subprocess, "Popen", lambda *a, **k: next(processes))
    monkeypatch.setattr(run_dev.sys, "executable", "python")
    monkeypatch.setattr(run_dev.time, "sleep", lambda _seconds: None)

    result = run_dev.main([])

    assert result == 1
    assert frontend.terminated is True


def test_run_dev_terminates_both_on_keyboard_interrupt(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    backend = FakeProcess(poll_results=[None])
    frontend = FakeProcess(poll_results=[None])
    processes = iter([backend, frontend])

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_dev.subprocess, "Popen", lambda *a, **k: next(processes))
    monkeypatch.setattr(run_dev.sys, "executable", "python")
    monkeypatch.setattr(run_dev.time, "sleep", interrupt)

    result = run_dev.main([])

    assert result == 0
    assert backend.terminated is True
    assert frontend.terminated is True


def test_run_dev_stop_process_kills_when_terminate_times_out() -> None:
    run_dev = _load_run_dev_module()
    stubborn = FakeProcess(poll_results=[None], wait_raises_timeout=True)

    run_dev._stop_process(stubborn)

    assert stubborn.terminated is True
    assert stubborn.killed is True


def test_run_dev_stops_backend_when_frontend_fails_to_start(monkeypatch) -> None:
    run_dev = _load_run_dev_module()
    backend = FakeProcess(poll_results=[None])
    calls = {"count": 0}

    def fake_popen(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return backend
        raise OSError("frontend failed to spawn")

    monkeypatch.setattr(run_dev.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_dev.sys, "executable", "python")

    try:
        run_dev.main([])
    except OSError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected OSError to propagate")

    assert backend.terminated is True
