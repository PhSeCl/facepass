import argparse
import os
import subprocess
import sys


MODEL_PATH_ENV_VAR = "FACEPASS_MODEL_PATH"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FacePass backend locally.")
    parser.add_argument(
        "--model-path",
        help="Path to a local buffalo_l model directory.",
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
    print("\n  FacePass → http://127.0.0.1:8000\n")
    try:
        backend.wait()
    except KeyboardInterrupt:
        backend.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
