"""Pop a native OS folder picker and return the chosen absolute path.

This is a local single-user convenience: the backend runs on the same desktop
session as the browser, so it can show the platform's native "选择文件夹" dialog
and hand the real absolute path back to the frontend (something a sandboxed
browser file input can never reveal).

The dialog runs in a separate subprocess on purpose. Tk insists on owning the
main thread of its process and runs its own event loop, which does not coexist
with the uvicorn worker threads. Isolating it in a child process keeps the
server process untouched and lets us bound the wait with a timeout.
"""

from __future__ import annotations

import subprocess
import sys


# Executed in a child Python process. Stays tiny and dependency-free (tkinter is
# stdlib). It writes the selected directory to stdout as raw UTF-8 bytes so that
# non-ASCII paths (e.g. Chinese folder names) survive regardless of the console
# code page. Empty output means the user cancelled.
_DIALOG_SCRIPT = r"""
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
selected = filedialog.askdirectory(title="选择数据集文件夹")
root.destroy()
sys.stdout.buffer.write((selected or "").encode("utf-8"))
"""


class DirectoryPickerUnavailable(RuntimeError):
    """Raised when the native folder dialog cannot be shown.

    Typically because there is no desktop session (headless/remote run) or
    tkinter is missing from the interpreter.
    """


def pick_directory_via_dialog(timeout: float = 300.0) -> str | None:
    """Show a native folder picker and return the chosen absolute path.

    Returns ``None`` if the user cancels. Raises ``DirectoryPickerUnavailable``
    if the dialog cannot be shown at all.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _DIALOG_SCRIPT],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DirectoryPickerUnavailable("选择文件夹对话框超时（5 分钟未操作）") from exc
    except OSError as exc:
        raise DirectoryPickerUnavailable("无法启动选择文件夹对话框") from exc

    if completed.returncode != 0:
        message = (
            completed.stderr.decode("utf-8", "replace").strip()
            or "无法打开选择文件夹对话框（可能没有图形界面或缺少 tkinter）"
        )
        raise DirectoryPickerUnavailable(message)

    path = completed.stdout.decode("utf-8", "replace").strip()
    return path or None
