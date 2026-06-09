import argparse
import importlib.util
import os
import subprocess
import sys


MODEL_PATH_ENV_VAR = "FACEPASS_MODEL_PATH"

# Mapping of import name -> pip package name for the libraries the backend needs
# to start. Used to give a clear "缺什么库" message before launching uvicorn.
REQUIRED_MODULES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "onnxruntime": "onnxruntime",
    "insightface": "insightface",
    "matplotlib": "matplotlib",
    "tomli_w": "tomli-w",
}
if sys.version_info < (3, 11):
    REQUIRED_MODULES["tomli"] = "tomli"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FacePass backend locally.")
    parser.add_argument(
        "--model-path",
        help="Path to a local buffalo_l model directory.",
    )
    return parser


def _missing_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(package_name)
    return missing


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    missing = _missing_dependencies()
    if missing:
        print("启动失败：缺少以下 Python 依赖库：")
        for package_name in missing:
            print(f"  - {package_name}")
        print()
        print("请先安装依赖再启动，例如：")
        print("  uv sync")
        print("  或: pip install " + " ".join(missing))
        return 2

    backend_env = os.environ.copy()
    if args.model_path:
        backend_env[MODEL_PATH_ENV_VAR] = args.model_path

    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.backend.api:app", "--port", "8000"],
        env=backend_env,
    )
    print("\n  FacePass → http://127.0.0.1:8000\n")
    try:
        # Propagate the backend's exit code so launchers (run.bat) can detect a
        # failed startup instead of always seeing success.
        return backend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
