from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from src.backend.dir_picker import (
    DirPickerError,
    DirPickerTkMissingError,
    pick_folder,
)


class TestPickFolder:
    def test_pick_folder_returns_path_on_success(self, tmp_path: Path) -> None:
        selected = tmp_path / "my_dataset"
        selected.mkdir()
        fake_result = mock.Mock()
        fake_result.returncode = 0
        fake_result.stdout = str(selected) + "\n"
        fake_result.stderr = ""

        with mock.patch("subprocess.run", return_value=fake_result):
            with mock.patch("src.backend.dir_picker.tkinter", create=True):
                result = pick_folder()

        assert result == selected.resolve()

    def test_pick_folder_raises_on_cancel(self) -> None:
        fake_result = mock.Mock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        fake_result.stderr = ""

        with mock.patch("subprocess.run", return_value=fake_result):
            with mock.patch("src.backend.dir_picker.tkinter", create=True):
                with pytest.raises(DirPickerError, match="用户取消了选择"):
                    pick_folder()

    def test_pick_folder_raises_on_empty_output(self) -> None:
        fake_result = mock.Mock()
        fake_result.returncode = 0
        fake_result.stdout = ""
        fake_result.stderr = ""

        with mock.patch("subprocess.run", return_value=fake_result):
            with mock.patch("src.backend.dir_picker.tkinter", create=True):
                with pytest.raises(DirPickerError, match="未选择任何文件夹"):
                    pick_folder()

    def test_pick_folder_raises_on_invalid_path(self) -> None:
        fake_result = mock.Mock()
        fake_result.returncode = 0
        fake_result.stdout = "/nonexistent/path\n"
        fake_result.stderr = ""

        with mock.patch("subprocess.run", return_value=fake_result):
            with mock.patch("src.backend.dir_picker.tkinter", create=True):
                with pytest.raises(DirPickerError, match="不是有效文件夹"):
                    pick_folder()

    def test_pick_folder_raises_stderr_on_failure(self) -> None:
        fake_result = mock.Mock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        fake_result.stderr = "Tcl error something"

        with mock.patch("subprocess.run", return_value=fake_result):
            with mock.patch("src.backend.dir_picker.tkinter", create=True):
                with pytest.raises(DirPickerError, match="Tcl error something"):
                    pick_folder()

    def test_pick_folder_raises_tkmissing_when_tkinter_not_importable(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "tkinter", None)
        with pytest.raises(DirPickerTkMissingError, match="未安装 tkinter"):
            pick_folder()


class TestPickDirectoryEndpoint:
    """Test /pick-directory using FastAPI TestClient with mocked pick_folder."""

    @pytest.fixture(autouse=True)
    def _patch_pick_folder(self, monkeypatch):
        self._mock_pick = mock.MagicMock()
        monkeypatch.setattr("src.backend.api.pick_folder", self._mock_pick)

    @pytest.fixture
    def client(self, tmp_path, monkeypatch) -> TestClient:
        from src.backend.api import app
        return TestClient(app)

    def test_pick_directory_returns_path(self, client) -> None:
        self._mock_pick.return_value = Path(r"C:/datasets/testdata")
        response = client.post("/pick-directory")
        assert response.status_code == 200
        body = response.json()
        assert body["path"] == str(Path(r"C:/datasets/testdata"))

    def test_pick_directory_returns_400_on_cancel(self, client) -> None:
        self._mock_pick.side_effect = DirPickerError("用户取消了选择")
        response = client.post("/pick-directory")
        assert response.status_code == 400
        body = response.json()
        assert "取消" in body.get("message", "")

    def test_pick_directory_returns_400_on_tkmissing(self, client) -> None:
        self._mock_pick.side_effect = DirPickerTkMissingError("未安装 tkinter")
        response = client.post("/pick-directory")
        assert response.status_code == 400
        body = response.json()
        assert "tkinter" in body.get("message", "").lower()
