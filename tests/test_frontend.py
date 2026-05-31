import requests
from PIL import Image

from src.frontend.app import recognize_via_backend


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
