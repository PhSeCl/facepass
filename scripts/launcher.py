"""FacePass smart launcher (stdlib-only).

Driven by run.bat. On first launch it runs a small wizard:

  1. Detect uv / python / virtualenv on the machine (silent).
  2. Pick an environment to run in (uv-managed .venv, a plain .venv, or global
     python), creating/initializing it on request.
  3. Check that the required packages are present *excluding onnxruntime*
     (CPU vs GPU is decided later), and offer a one-click sync.
  4. Ask CPU or GPU. For GPU, set up onnxruntime-gpu and verify CUDA. For CPU,
     make sure a CPU onnxruntime is present.
  5. Persist the choice to config.toml [runtime] and launch scripts/run_dev.py
     with the right command (uv run / .venv python / global python).

On later launches the saved [runtime] config is used directly; if that
environment no longer validates (missing venv / deps / CUDA), it falls back to
re-running the wizard.

Kept stdlib-only on purpose: this runs under whatever interpreter run.bat could
find (possibly bare global python), so it must not import project deps. config
read/write is done by light text surgery on the [runtime] table, preserving
every other section.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
RUN_DEV = REPO_ROOT / "scripts" / "run_dev.py"
VENV_PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
ORT_PACKAGE_DIR = REPO_ROOT / ".venv" / "Lib" / "site-packages" / "onnxruntime"

# Dedicated GPU virtualenv: kept separate from the project .venv so that
# `uv run`/`uv sync` (which re-sync the CPU-pinned onnxruntime) never clobber the
# GPU build. Installed once, then reused on every later GPU launch.
GPU_VENV = REPO_ROOT / ".venv-gpu"
GPU_VENV_PY = GPU_VENV / "Scripts" / "python.exe"
GPU_ORT_PACKAGE_DIR = GPU_VENV / "Lib" / "site-packages" / "onnxruntime"

CPU_ONNXRUNTIME = "onnxruntime==1.22.1"
GPU_ONNXRUNTIME = "onnxruntime-gpu[cuda,cudnn]"


# --------------------------------------------------------------------------- #
# Required modules (reuse run_dev's list, minus onnxruntime which is decided
# only after the CPU/GPU question).
# --------------------------------------------------------------------------- #
def required_modules() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("facepass_run_dev", RUN_DEV)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return {name: pkg for name, pkg in module.REQUIRED_MODULES.items() if name != "onnxruntime"}


# --------------------------------------------------------------------------- #
# config.toml [runtime] read/write (text surgery, no TOML dependency).
# --------------------------------------------------------------------------- #
def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    return value


def read_runtime(text: str) -> dict[str, str] | None:
    """Parse the [runtime] table out of config.toml text, or None if absent."""
    lines = text.splitlines()
    in_section = False
    result: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped[1:-1].strip() == "runtime"
            continue
        if in_section and "=" in stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition("=")
            result[key.strip()] = _strip_quotes(value)
    return result or None


def strip_runtime(text: str) -> str:
    """Remove an existing [runtime] table, leaving every other section intact."""
    out: list[str] = []
    skipping = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            skipping = stripped[1:-1].strip() == "runtime"
        if not skipping:
            out.append(line)
    cleaned = "\n".join(out).rstrip("\n")
    return cleaned


def save_runtime(device: str, launcher: str) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
    body = strip_runtime(text)
    block = f'[runtime]\ndevice = "{device}"\nlauncher = "{launcher}"\n'
    if body:
        body = body + "\n\n" + block
    else:
        body = block
    CONFIG_PATH.write_text(body, encoding="utf-8")


def load_runtime() -> dict[str, str] | None:
    if not CONFIG_PATH.exists():
        return None
    return read_runtime(CONFIG_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Interpreter resolution.
# --------------------------------------------------------------------------- #
def resolve_launch_command(launcher: str) -> list[str] | None:
    """Map a launcher token to the command used to start run_dev.py."""
    if launcher == "uv":
        return ["uv", "run", "python"]
    if launcher == "venv":
        return [str(VENV_PY)]
    if launcher == "venv-gpu":
        return [str(GPU_VENV_PY)]
    if launcher == "global":
        python = shutil.which("python") or (sys.executable if sys.executable else None)
        return [python] if python else None
    return None


def _check_interpreter(launcher: str) -> str | None:
    """The concrete interpreter path used for package/CUDA introspection.

    Always a real interpreter (never `uv run`), so probing never triggers a
    lockfile re-sync.
    """
    if launcher in ("uv", "venv"):
        return str(VENV_PY) if VENV_PY.exists() else None
    if launcher == "venv-gpu":
        return str(GPU_VENV_PY) if GPU_VENV_PY.exists() else None
    if launcher == "global":
        return shutil.which("python")
    return None


# --------------------------------------------------------------------------- #
# Probing helpers (run in the *target* interpreter via subprocess).
# --------------------------------------------------------------------------- #
def missing_packages(interpreter: str) -> list[str] | None:
    """pip names of required modules missing in `interpreter` (onnxruntime excluded).

    Returns None if the probe itself could not run.
    """
    modules = required_modules()
    code = (
        "import importlib.util as u\n"
        f"m={list(modules.keys())!r}\n"
        "print(','.join(x for x in m if u.find_spec(x) is None))\n"
    )
    try:
        proc = subprocess.run(
            [interpreter, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    found_missing = [name for name in proc.stdout.strip().split(",") if name]
    return [modules[name] for name in found_missing]


def _probe_import(interpreter: str, module: str) -> bool:
    try:
        proc = subprocess.run(
            [interpreter, "-c", f"import importlib.util as u,sys; sys.exit(0 if u.find_spec({module!r}) else 1)"],
            capture_output=True,
            cwd=str(REPO_ROOT),
        )
    except OSError:
        return False
    return proc.returncode == 0


def has_onnxruntime(interpreter: str) -> bool:
    return _probe_import(interpreter, "onnxruntime")


def cuda_available(interpreter: str) -> bool:
    code = (
        "import onnxruntime as o,sys; "
        "sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)"
    )
    try:
        proc = subprocess.run([interpreter, "-c", code], capture_output=True, cwd=str(REPO_ROOT))
    except OSError:
        return False
    return proc.returncode == 0


# --------------------------------------------------------------------------- #
# Prompts.
# --------------------------------------------------------------------------- #
def ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            answer = input(question + suffix).strip().lower()
        except EOFError:
            return default
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("请输入 y 或 n。")


def ask_device() -> str:
    while True:
        try:
            answer = input("使用 CPU 还是 GPU 运行? [C/g] ").strip().lower()
        except EOFError:
            return "cpu"
        if answer in ("", "c", "cpu"):
            return "cpu"
        if answer in ("g", "gpu"):
            return "gpu"
        print("请输入 c 或 g。")


# --------------------------------------------------------------------------- #
# Dependency / onnxruntime setup.
# --------------------------------------------------------------------------- #
def sync_dependencies(launcher: str, interpreter: str) -> None:
    if launcher == "uv":
        subprocess.run(["uv", "sync"], cwd=str(REPO_ROOT))
    else:
        subprocess.run([interpreter, "-m", "pip", "install", "-r", str(REQUIREMENTS)], cwd=str(REPO_ROOT))


def ensure_cpu_onnxruntime(launcher: str, interpreter: str) -> None:
    if launcher == "uv":
        # `uv run` will sync the locked CPU onnxruntime at launch time.
        return
    if not has_onnxruntime(interpreter):
        print("安装 CPU 版 onnxruntime ...")
        subprocess.run([interpreter, "-m", "pip", "install", CPU_ONNXRUNTIME], cwd=str(REPO_ROOT))


def _pip_uninstall(target_py: str, use_uv: bool, packages: list[str]) -> None:
    if use_uv:
        subprocess.run(["uv", "pip", "uninstall", "--python", target_py, *packages], cwd=str(REPO_ROOT))
    else:
        subprocess.run([target_py, "-m", "pip", "uninstall", "-y", *packages], cwd=str(REPO_ROOT))


def _pip_install(target_py: str, use_uv: bool, args: list[str]) -> int:
    if use_uv:
        return subprocess.run(["uv", "pip", "install", "--python", target_py, *args], cwd=str(REPO_ROOT)).returncode
    return subprocess.run([target_py, "-m", "pip", "install", *args], cwd=str(REPO_ROOT)).returncode


def install_gpu_onnxruntime(target_py: str, use_uv: bool, ort_pkg_dir: Path | None) -> bool:
    """Install onnxruntime-gpu into `target_py` and verify CUDA. Returns success."""
    if cuda_available(target_py):
        return True
    print("配置 GPU 版 onnxruntime（下载/解压 CUDA、cuDNN；首次较久，之后从 pip/uv 缓存复用）...")
    _pip_uninstall(target_py, use_uv, ["onnxruntime", "onnxruntime-gpu"])
    # Clear a leftover empty onnxruntime dir (a known half-uninstall state).
    if ort_pkg_dir is not None and ort_pkg_dir.exists():
        shutil.rmtree(ort_pkg_dir, ignore_errors=True)
    force = "--reinstall" if use_uv else "--force-reinstall"
    if _pip_install(target_py, use_uv, [force, GPU_ONNXRUNTIME]) != 0:
        print("[错误] 安装 onnxruntime-gpu 失败。")
        return False
    if not cuda_available(target_py):
        print("[错误] 安装后仍检测不到 CUDAExecutionProvider，请检查 NVIDIA 驱动 / GPU。")
        return False
    return True


def create_gpu_venv(has_uv: bool, base_interpreter: str | None) -> bool:
    """Create the dedicated .venv-gpu (via uv, else the base python's venv module)."""
    if GPU_VENV_PY.exists():
        return True
    print(f"创建 GPU 专用虚拟环境 {GPU_VENV.name} ...")
    if has_uv:
        rc = subprocess.run(["uv", "venv", str(GPU_VENV)], cwd=str(REPO_ROOT)).returncode
    elif base_interpreter:
        rc = subprocess.run([base_interpreter, "-m", "venv", str(GPU_VENV)], cwd=str(REPO_ROOT)).returncode
    else:
        return False
    return rc == 0 and GPU_VENV_PY.exists()


def _install_requirements_without_onnxruntime(target_py: str, use_uv: bool) -> bool:
    """Install requirements.txt minus onnxruntime (the GPU build is added separately)."""
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not ln.strip().lower().startswith("onnxruntime")]
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        handle.write("\n".join(kept) + "\n")
        handle.close()
        return _pip_install(target_py, use_uv, ["-r", handle.name]) == 0
    finally:
        os.unlink(handle.name)


def setup_gpu_runtime(has_uv: bool, base_launcher: str, base_interpreter: str) -> tuple[str, str] | None:
    """Prepare a GPU runtime. Returns (launch_token, device) or None to abort.

    Preferred path: a dedicated .venv-gpu that `uv` never re-syncs, so the GPU
    build is installed once and reused on every later launch (no re-download,
    even after running CPU mode). If that environment cannot be built, falls back
    to an interactive choice.
    """
    gpu_py = str(GPU_VENV_PY)
    # Reuse a healthy dedicated env (the common case after first setup).
    if GPU_VENV_PY.exists():
        missing = missing_packages(gpu_py)
        if missing is not None and not missing and cuda_available(gpu_py):
            print("[OK] 复用已有 GPU 专用环境 .venv-gpu（无需重装）。")
            return "venv-gpu", "gpu"
    # Build / repair the dedicated env.
    if create_gpu_venv(has_uv, base_interpreter):
        print("向 .venv-gpu 安装依赖 ...")
        if _install_requirements_without_onnxruntime(gpu_py, has_uv) and install_gpu_onnxruntime(
            gpu_py, has_uv, GPU_ORT_PACKAGE_DIR
        ):
            print("[OK] GPU 专用环境 .venv-gpu 就绪。")
            return "venv-gpu", "gpu"
        print("[警告] 配置 .venv-gpu 失败。")
    else:
        print("[警告] 无法创建 GPU 专用虚拟环境 .venv-gpu。")
    return _gpu_fallback_choice(has_uv, base_launcher, base_interpreter)


def _gpu_fallback_choice(has_uv: bool, base_launcher: str, base_interpreter: str) -> tuple[str, str] | None:
    print()
    print("无法建立 GPU 专用环境（.venv-gpu）。可能原因：没有 uv、当前 python 缺少 venv/pip，或权限/磁盘问题。")
    print("请选择如何处理：")
    print("  [1] 把 onnxruntime-gpu 直接装进当前环境（可用，但之后切到 CPU 会覆盖它、再切回 GPU 需重装）")
    print("  [2] 回退到 CPU 运行")
    print("  [3] 放弃，退出")
    while True:
        try:
            answer = input("输入 1 / 2 / 3: ").strip()
        except EOFError:
            return None
        if answer == "1":
            use_uv_pip = base_launcher == "uv"
            ort_dir = ORT_PACKAGE_DIR if base_interpreter == str(VENV_PY) else None
            if install_gpu_onnxruntime(base_interpreter, use_uv_pip, ort_dir):
                # Launch via the interpreter directly (never `uv run`) so the
                # just-installed GPU build is not immediately re-synced away.
                return ("global" if base_launcher == "global" else "venv"), "gpu"
            print("[错误] 在当前环境安装 GPU 失败。")
            return None
        if answer == "2":
            return base_launcher, "cpu"
        if answer == "3":
            return None
        print("请输入 1、2 或 3。")


# --------------------------------------------------------------------------- #
# Wizard.
# --------------------------------------------------------------------------- #
def choose_environment(has_uv: bool) -> tuple[str, str] | None:
    """Return (launcher_token, interpreter_path) or None if no env can serve."""
    # ---- uv examination ----
    if has_uv:
        if VENV_PY.exists():
            print("[OK] 检测到 uv，且项目已有虚拟环境 .venv。")
            return "uv", str(VENV_PY)
        print("[INFO] 检测到 uv，但项目尚未初始化虚拟环境。")
        if ask_yes_no("是否用 uv 初始化项目环境 (uv sync)?"):
            rc = subprocess.run(["uv", "sync"], cwd=str(REPO_ROOT)).returncode
            if rc == 0 and VENV_PY.exists():
                return "uv", str(VENV_PY)
            print("[警告] uv sync 未成功，转为考察 python 环境。")

    # ---- python examination ----
    if VENV_PY.exists():
        print("[INFO] 检测到项目虚拟环境 .venv。")
        if ask_yes_no("是否使用这个虚拟环境?"):
            return "venv", str(VENV_PY)

    global_python = shutil.which("python")
    if global_python:
        if ask_yes_no("是否使用全局 python?", default=False):
            return "global", global_python
        print("[退出] 已选择不使用全局 python。")
        return None

    print("[错误] 未找到可用的 uv / python，无法提供服务。")
    return None


def run_wizard(has_uv: bool) -> tuple[str, str] | None:
    """Run the interactive wizard. Returns (launcher, device) or None on abort."""
    print("==============================================")
    print("  FacePass 启动向导")
    print("==============================================")

    chosen = choose_environment(has_uv)
    if chosen is None:
        return None
    launcher, interpreter = chosen

    # Dependency check (excluding onnxruntime).
    missing = missing_packages(interpreter)
    if missing is None:
        print("[警告] 无法在所选环境中检测依赖，跳过依赖检查。")
    elif missing:
        print("[INFO] 检测到缺失依赖（不含 onnxruntime）：")
        for pkg in missing:
            print("        -", pkg)
        if ask_yes_no("是否一键同步依赖?"):
            sync_dependencies(launcher, interpreter)
    else:
        print("[OK] 依赖完整（onnxruntime 待 CPU/GPU 选择后处理）。")

    # CPU / GPU.
    device = ask_device()
    if device == "gpu":
        result = setup_gpu_runtime(has_uv, launcher, interpreter)
        if result is None:
            return None
        launch_token, device = result
        if device == "cpu":
            # User chose the CPU fallback when no GPU env could be built.
            ensure_cpu_onnxruntime(launch_token, interpreter)
    else:
        ensure_cpu_onnxruntime(launcher, interpreter)
        launch_token = launcher

    save_runtime(device, launch_token)
    print(f"[OK] 已保存运行配置到 config.toml: device={device}, launcher={launch_token}")
    return launch_token, device


# --------------------------------------------------------------------------- #
# Launch.
# --------------------------------------------------------------------------- #
def launch(launcher: str, device: str) -> int:
    command = resolve_launch_command(launcher)
    if command is None:
        print(f"[错误] 无法解析启动命令: launcher={launcher}")
        return 1
    print(f"启动 FacePass [{device.upper()}] (launcher={launcher}) ...")
    print()
    try:
        return subprocess.run(command + [str(RUN_DEV)], cwd=str(REPO_ROOT)).returncode
    except OSError as exc:
        print(f"[错误] 启动失败: {exc}")
        return 1


def runtime_is_valid(runtime: dict[str, str], has_uv: bool) -> bool:
    launcher = runtime.get("launcher")
    device = runtime.get("device")
    if launcher not in ("uv", "venv", "venv-gpu", "global") or device not in ("cpu", "gpu"):
        return False
    if launcher == "uv" and not has_uv:
        return False
    interpreter = _check_interpreter(launcher)
    if not interpreter:
        return False
    missing = missing_packages(interpreter)
    if missing is None or missing:
        return False
    if not has_onnxruntime(interpreter):
        return False
    if device == "gpu" and not cuda_available(interpreter):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FacePass smart launcher.")
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Ignore the saved [runtime] config and re-run the wizard.",
    )
    args = parser.parse_args(argv)

    has_uv = shutil.which("uv") is not None

    runtime = None if args.reconfigure else load_runtime()
    if runtime is not None:
        if runtime_is_valid(runtime, has_uv):
            print(
                f"[OK] 使用已保存的运行配置: device={runtime['device']}, "
                f"launcher={runtime['launcher']}（如需重选: run.bat --reconfigure）"
            )
            return launch(runtime["launcher"], runtime["device"])
        print("[INFO] 已保存的运行配置已失效，重新检测环境...")

    result = run_wizard(has_uv)
    if result is None:
        try:
            input("按回车键退出...")
        except EOFError:
            pass
        return 1
    launcher, device = result
    return launch(launcher, device)


if __name__ == "__main__":
    raise SystemExit(main())
