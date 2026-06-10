"""Open a native folder-picker dialog in a subprocess, returning the chosen path.

Uses tkinter's askdirectory, isolated in a child Python process so that Tk's
event loop never collides with uvicorn's async runtime.  No extra dependencies
are needed (tkinter ships with CPython on Windows, macOS, and most Linux
distributions).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_PICKER_SCRIPT = r"""
import sys, tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
path = filedialog.askdirectory(title='选择数据集文件夹')
root.destroy()
if path:
    print(path, flush=True)
else:
    sys.exit(1)
"""


class DirPickerError(Exception):
    """Raised when the folder picker cannot be used or the user cancels."""


class DirPickerTkMissingError(DirPickerError):
    """Raised when tkinter is not available in the runtime environment."""


def pick_folder() -> Path:
    """Open a native folder-picker dialog and return the selected directory.

    Returns:
        Path: absolute path to the selected directory.

    Raises:
        DirPickerTkMissingError: tkinter is not available (e.g. headless server).
        DirPickerError: user cancelled or an unexpected error occurred.
    """
    try:
        import tkinter  # noqa: F401  – verify importable
    except ImportError as exc:
        raise DirPickerTkMissingError("当前环境未安装 tkinter，无法打开文件夹选择对话框") from exc

    executable = sys.executable
    result = subprocess.run(
        [executable, "-c", _PICKER_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            raise DirPickerError(f"文件夹选择失败: {stderr}")
        raise DirPickerError("用户取消了选择")

    selected = result.stdout.strip()
    if not selected:
        raise DirPickerError("未选择任何文件夹")

    path = Path(selected).resolve()
    if not path.is_dir():
        raise DirPickerError(f"选择的路径不是有效文件夹: {selected}")

    return path
