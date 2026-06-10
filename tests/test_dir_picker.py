import subprocess
from types import SimpleNamespace

import pytest

import src.backend.dir_picker as dir_picker
from src.backend.dir_picker import DirectoryPickerUnavailable, pick_directory_via_dialog


def _fake_completed(returncode: int, stdout: bytes = b"", stderr: bytes = b""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_pick_directory_decodes_utf8_path(monkeypatch) -> None:
    # The child writes raw UTF-8 bytes; non-ASCII (Chinese) paths must survive.
    path = "F:/数据集/test"
    monkeypatch.setattr(
        dir_picker.subprocess,
        "run",
        lambda *args, **kwargs: _fake_completed(0, stdout=path.encode("utf-8")),
    )

    assert pick_directory_via_dialog() == path


def test_pick_directory_returns_none_when_cancelled(monkeypatch) -> None:
    monkeypatch.setattr(
        dir_picker.subprocess,
        "run",
        lambda *args, **kwargs: _fake_completed(0, stdout=b"   "),
    )

    assert pick_directory_via_dialog() is None


def test_pick_directory_raises_on_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr(
        dir_picker.subprocess,
        "run",
        lambda *args, **kwargs: _fake_completed(1, stderr="缺少 tkinter".encode("utf-8")),
    )

    with pytest.raises(DirectoryPickerUnavailable, match="缺少 tkinter"):
        pick_directory_via_dialog()


def test_pick_directory_raises_on_timeout(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=1)

    monkeypatch.setattr(dir_picker.subprocess, "run", fake_run)

    with pytest.raises(DirectoryPickerUnavailable, match="超时"):
        pick_directory_via_dialog()
