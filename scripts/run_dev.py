import argparse
import os
import subprocess
import sys
import time


MODEL_PATH_ENV_VAR = "FACEPASS_MODEL_PATH"
POLL_INTERVAL_SECONDS = 0.5
TERMINATE_TIMEOUT_SECONDS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FacePass backend and frontend locally.")
    parser.add_argument(
        "--model-path",
        help="Path to a local buffalo_l model directory passed to the backend at startup.",
    )
    return parser


def _stop_process(process: subprocess.Popen) -> None:
    """Terminate a child process, escalating to kill if it ignores terminate."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _supervise(processes: dict[str, subprocess.Popen]) -> int:
    """Block until any child exits or Ctrl+C, then tear the rest down.

    Keeping backend and frontend tied together avoids leaving one half of the
    app running as an orphan when the other crashes.
    """
    try:
        while True:
            for name, process in processes.items():
                returncode = process.poll()
                if returncode is not None:
                    print(
                        f"[run_dev] {name} 进程已退出 (returncode={returncode})，正在关闭其余进程",
                        file=sys.stderr,
                    )
                    return returncode or 0
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("[run_dev] 收到中断信号，正在关闭后端与前端", file=sys.stderr)
        return 0
    finally:
        for process in processes.values():
            _stop_process(process)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend_env = os.environ.copy()
    if args.model_path:
        backend_env[MODEL_PATH_ENV_VAR] = args.model_path

    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.backend.api:app", "--port", "8000"],
        env=backend_env,
    )
    try:
        frontend = subprocess.Popen([sys.executable, "src/frontend/app.py"])
    except BaseException:
        # Never leave a started backend running if the frontend fails to spawn.
        _stop_process(backend)
        raise

    return _supervise({"backend": backend, "frontend": frontend})


if __name__ == "__main__":
    raise SystemExit(main())
