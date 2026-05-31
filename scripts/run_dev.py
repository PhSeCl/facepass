import argparse
import os
import subprocess
import sys


MODEL_PATH_ENV_VAR = "FACEPASS_MODEL_PATH"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FacePass backend and frontend locally.")
    parser.add_argument(
        "--model-path",
        help="Path to a local buffalo_l model directory passed to the backend at startup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend_env = os.environ.copy()
    if args.model_path:
        backend_env[MODEL_PATH_ENV_VAR] = args.model_path

    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.backend.api:app", "--port", "8000"],
        env=backend_env,
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
