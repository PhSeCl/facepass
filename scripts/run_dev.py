import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


MODEL_PATH_ENV_VAR = "FACEPASS_MODEL_PATH"
REPO_ROOT = Path(__file__).resolve().parents[1]

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


# Indirection so tests can drive the prompts; also lets EOFError be handled in one
# place. input() is only ever reached on the interactive (double-click) path.
def _prompt(text: str) -> str:
    return input(text)


def _ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            answer = _prompt(question + suffix).strip().lower()
        except EOFError:
            return default
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("请输入 y 或 n。")


def _ensure_repo_on_path() -> None:
    # Running as `python scripts/run_dev.py` puts scripts/ (not the repo root) on
    # sys.path[0], so `import src.*` would fail without this.
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_model_path_interactively() -> tuple[str, str | None]:
    """Resolve a usable model path for the double-click / launcher flow.

    Returns an (action, value) pair:
      ("config", None) — a valid path is already in config.toml; the backend
                         reads it, nothing to pass.
      ("env", path)    — use this path for this session only (user declined to
                         save it; it is not written to config.toml).
      ("abort", None)  — the user gave no path; FacePass cannot start.

    Deps are imported lazily, after the dependency check in main(), because
    model_config pulls in third-party packages (tomli_w) the bootstrap stage
    cannot assume are present.
    """
    _ensure_repo_on_path()
    from src.common.errors import ModelConfigError
    from src.face_model import model_config

    # A previously saved, still-valid path needs no interaction.
    persisted = model_config.load_persisted_path()
    if persisted is not None:
        try:
            model_config.validate_model_path(persisted)
            return "config", None
        except ModelConfigError as exc:
            print(f"[警告] config.toml 中已保存的模型路径已失效：{exc}")
            print("       请重新提供一个有效的模型目录。")

    print()
    print("未找到可用的模型路径。FacePass 需要一个本地 buffalo_l 模型目录")
    print("（该目录直接包含 det_10g.onnx、w600k_r50.onnx 等 5 个 .onnx 文件）。")
    print("请将该目录路径粘贴到下面（留空并回车则退出）。")
    while True:
        try:
            raw = _prompt("模型目录路径: ").strip().strip('"').strip("'")
        except EOFError:
            return "abort", None
        if not raw:
            return "abort", None
        try:
            validated = model_config.validate_model_path(raw)
        except ModelConfigError as exc:
            print(f"[校验失败] {exc}")
            print("          请重新粘贴正确的 buffalo_l 目录，或留空回车退出。")
            continue
        print(f"[OK] 模型校验通过：{validated}")
        if _ask_yes_no("是否将该路径保存为默认模型路径（下次启动不再询问）?", default=True):
            model_config.persist_path(validated)
            print("[OK] 已保存到 config.toml，下次启动将直接使用。")
            return "config", None
        print("[OK] 本次启动使用该路径（未保存，下次启动仍会询问）。")
        return "env", str(validated)


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
        # Explicit CLI override: pass through unchanged (developer escape hatch).
        backend_env[MODEL_PATH_ENV_VAR] = args.model_path
    else:
        # Double-click / launcher path: resolve the model path interactively
        # instead of letting the backend hard-exit when it is missing.
        action, value = _resolve_model_path_interactively()
        if action == "abort":
            print()
            print("已取消：未提供模型路径，无法启动 FacePass。")
            print("可重新双击 run.bat，或参考 README「配置模型路径」一节。")
            return 1
        if action == "env" and value is not None:
            backend_env[MODEL_PATH_ENV_VAR] = value
        # action == "config": leave the env var unset; the backend reads config.toml.

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
