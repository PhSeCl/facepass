import subprocess
import sys


def main() -> int:
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.backend.api:app", "--port", "8000"]
    )
    frontend = subprocess.Popen([sys.executable, "src/frontend/app.py"])
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        frontend.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
