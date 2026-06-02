import requests
import pytest
from PIL import Image
from pathlib import Path

from src.frontend.app import (
    DATASET_BROWSER_ROOT,
    DATASET_EVAL_TIMEOUT,
    demo,
    inspect_dataset_via_backend,
    _post_dataset_eval_directory,
    _post_dataset_eval,
    _post_dataset_inspect_directory,
    recognize_via_backend,
    run_dataset_eval_via_backend,
)


def test_frontend_returns_friendly_error_when_backend_unreachable(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)

    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("src.frontend.app.requests.post", raise_connection_error)

    annotated, rows, message = recognize_via_backend(str(image_path), backend_url="http://127.0.0.1:9")

    assert annotated is None
    assert rows == []
    assert "后端未启动" in message


def test_frontend_inspects_dataset_and_shows_gallery_choice(monkeypatch, tmp_path) -> None:
    archive_path = tmp_path / "test.zip"
    archive_path.write_bytes(b"zip")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"has_registered": True}

    monkeypatch.setattr("src.frontend.app.requests.post", lambda *args, **kwargs: FakeResponse())

    has_registered, message = inspect_dataset_via_backend(str(archive_path), backend_url="http://127.0.0.1:8000")

    assert has_registered is True
    assert "registered/" in message


def test_frontend_inspects_dataset_directory_and_shows_gallery_choice(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()

    class FakeResponse:
        def json(self) -> dict:
            return {"has_registered": True}

    monkeypatch.setattr("src.frontend.app._post_dataset_inspect_directory", lambda *args, **kwargs: FakeResponse())

    has_registered, message = inspect_dataset_via_backend(
        str(dataset_dir),
        source_mode="directory",
        backend_url="http://127.0.0.1:8000",
    )

    assert has_registered is True
    assert "文件夹内含" in message


def test_frontend_evaluates_dataset_and_decodes_plot_images(monkeypatch, tmp_path) -> None:
    archive_path = tmp_path / "test.zip"
    archive_path.write_bytes(b"zip")
    data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    class FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "gallery_source": "archive",
                "metrics": {
                    "strict_top1_accuracy": 1.0,
                    "matched_top1_accuracy": 0.9,
                    "detection_recall": 0.8,
                    "detection_precision": 0.7,
                    "unknown_detected_accuracy": 0.6,
                    "predicted_unknown_precision": 0.5,
                },
                "plots": {
                    "confusion_matrix": data_url,
                    "detection_metrics": data_url,
                    "accuracy_metrics": data_url,
                },
                "missed_detections": [{"image_name": "miss.jpg", "bbox": [1, 2, 3, 4]}],
                "false_positives": [{"image_name": "fp.jpg", "bbox": [5, 6, 7, 8]}],
            }

    monkeypatch.setattr("src.frontend.app._post_dataset_eval", lambda *args, **kwargs: FakeResponse())

    (
        summary,
        confusion_html,
        detection_html,
        accuracy_html,
        missed_rows,
        false_rows,
        message,
    ) = run_dataset_eval_via_backend(str(archive_path), "archive", backend_url="http://127.0.0.1:8000")

    assert "strict top-1" in summary
    assert 'src="data:image/png;base64,' in confusion_html
    assert 'src="data:image/png;base64,' in detection_html
    assert 'src="data:image/png;base64,' in accuracy_html
    assert missed_rows == [["miss.jpg", "1, 2, 3, 4"]]
    assert false_rows == [["fp.jpg", "5, 6, 7, 8"]]
    assert message == ""


def test_frontend_evaluates_dataset_directory_and_decodes_plot_images(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()
    data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    class FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "gallery_source": "local",
                "metrics": {
                    "strict_top1_accuracy": 1.0,
                    "matched_top1_accuracy": 0.9,
                    "detection_recall": 0.8,
                    "detection_precision": 0.7,
                    "unknown_detected_accuracy": 0.6,
                    "predicted_unknown_precision": 0.5,
                },
                "plots": {
                    "confusion_matrix": data_url,
                    "detection_metrics": data_url,
                    "accuracy_metrics": data_url,
                },
                "missed_detections": [],
                "false_positives": [],
            }

    monkeypatch.setattr("src.frontend.app._post_dataset_eval_directory", lambda *args, **kwargs: FakeResponse())

    summary, confusion_html, detection_html, accuracy_html, missed_rows, false_rows, message = (
        run_dataset_eval_via_backend(
            str(dataset_dir),
            "local",
            source_mode="directory",
            backend_url="http://127.0.0.1:8000",
        )
    )

    assert "strict top-1" in summary
    assert 'src="data:image/png;base64,' in confusion_html
    assert 'src="data:image/png;base64,' in detection_html
    assert 'src="data:image/png;base64,' in accuracy_html
    assert missed_rows == []
    assert false_rows == []
    assert message == ""


def test_frontend_dataset_eval_uses_longer_timeout(monkeypatch, tmp_path) -> None:
    archive_path = tmp_path / "test.zip"
    archive_path.write_bytes(b"zip")
    observed: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

    def fake_post(*args, **kwargs):
        observed["timeout"] = kwargs.get("timeout")
        return FakeResponse()

    monkeypatch.setattr("src.frontend.app.requests.post", fake_post)

    _post_dataset_eval(str(archive_path), "local", backend_url="http://127.0.0.1:8000")

    assert observed["timeout"] == DATASET_EVAL_TIMEOUT


def test_frontend_dataset_directory_eval_uses_longer_timeout(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()
    observed: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

    def fake_post(*args, **kwargs):
        observed["timeout"] = kwargs.get("timeout")
        observed["data"] = kwargs.get("data")
        return FakeResponse()

    monkeypatch.setattr("src.frontend.app.requests.post", fake_post)

    _post_dataset_eval_directory(str(dataset_dir), "local", backend_url="http://127.0.0.1:8000")

    assert observed["timeout"] == DATASET_EVAL_TIMEOUT
    assert observed["data"] == {"gallery_choice": "local", "dataset_dir": str(dataset_dir)}


def test_frontend_dataset_directory_inspect_posts_dataset_dir(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()
    observed: dict[str, object] = {}

    class FakeResponse:
        pass

    def fake_post(*args, **kwargs):
        observed["data"] = kwargs.get("data")
        observed["timeout"] = kwargs.get("timeout")
        return FakeResponse()

    monkeypatch.setattr("src.frontend.app.requests.post", fake_post)

    _post_dataset_inspect_directory(str(dataset_dir), backend_url="http://127.0.0.1:8000")

    assert observed["data"] == {"dataset_dir": str(dataset_dir)}
    assert observed["timeout"] == 5


def test_frontend_directory_browser_defaults_to_workspace_root() -> None:
    assert DATASET_BROWSER_ROOT == str(Path.cwd())


def test_frontend_directory_browser_does_not_auto_inspect_on_change() -> None:
    components = {component["id"]: component for component in demo.config["components"]}
    directory_component_id = next(
        component_id
        for component_id, component in components.items()
        if component["type"] == "fileexplorer"
        and component["props"].get("label") == "选择数据集文件夹"
    )

    assert all(
        not any(target == [directory_component_id, "change"] for target in dependency["targets"])
        for dependency in demo.config["dependencies"]
    )


def test_frontend_dataset_eval_timeout_returns_friendly_message(monkeypatch, tmp_path) -> None:
    archive_path = tmp_path / "test.zip"
    archive_path.write_bytes(b"zip")

    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("read timed out")

    monkeypatch.setattr("src.frontend.app._post_dataset_eval", raise_timeout)

    result = run_dataset_eval_via_backend(str(archive_path), "local", backend_url="http://127.0.0.1:8000")

    assert result == ("", "", "", "", [], [], "数据集评测超时，请稍后重试或检查后端日志")


def test_frontend_dataset_eval_timeout_does_not_retry_heavy_request(monkeypatch, tmp_path) -> None:
    archive_path = tmp_path / "test.zip"
    archive_path.write_bytes(b"zip")
    attempts = {"count": 0}

    def raise_timeout(*args, **kwargs):
        attempts["count"] += 1
        raise requests.Timeout("read timed out")

    monkeypatch.setattr("src.frontend.app.requests.post", raise_timeout)

    with pytest.raises(requests.Timeout):
        _post_dataset_eval(str(archive_path), "local", backend_url="http://127.0.0.1:8000")

    assert attempts["count"] == 1
