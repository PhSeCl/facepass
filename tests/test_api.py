import logging
from fastapi.testclient import TestClient
from types import SimpleNamespace
from PIL import Image
import io

import pytest

import src.backend.api as api
from src.common.errors import ModelLoadError, ModelNotFoundError, ModelPathMissingError

app = api.app


def test_health_endpoint_reports_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_identities_endpoint_returns_identity_counts() -> None:
    client = TestClient(app)

    response = client.get("/identities")

    assert response.status_code == 200
    assert "identities" in response.json()


def test_recognize_rejects_non_image_upload_with_400() -> None:
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert "message" in response.json()


def test_recognize_accepts_valid_image_upload() -> None:
    client = TestClient(app)
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buffer, format="PNG")

    response = client.post(
        "/recognize",
        files={"file": ("ok.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_load_identities_uses_current_settings_path(monkeypatch, tmp_path) -> None:
    identities_csv = tmp_path / "identities.csv"
    identities_csv.write_text("identity_id,name\np01,Alice\n", encoding="utf-8")
    monkeypatch.setattr(api, "settings", SimpleNamespace(identities_csv=identities_csv))

    assert api.load_identities() == {"p01": "Alice"}


def test_startup_passes_explicit_model_path_to_insightface_factory(monkeypatch, tmp_path) -> None:
    gallery_path = tmp_path / "gallery.pkl"
    gallery_path.write_bytes(b"placeholder")
    identities_csv = tmp_path / "identities.csv"
    identities_csv.write_text("identity_id,name\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_create_model(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        api,
        "settings",
        SimpleNamespace(
            model_name="insightface",
            threshold=0.3,
            gallery_path=gallery_path,
            registered_dir=tmp_path / "registered",
            identities_csv=identities_csv,
            max_upload_bytes=1024,
        ),
    )
    monkeypatch.setattr(api, "create_model", fake_create_model)
    monkeypatch.setattr(api.Gallery, "load", staticmethod(lambda path: api.Gallery()))
    monkeypatch.setattr(api, "Recognizer", lambda model, gallery, threshold, id2name: "recognizer")
    monkeypatch.setattr(api, "_recognizer", None)
    monkeypatch.setattr(api, "_gallery", api.Gallery())
    monkeypatch.setattr(api, "_id2name", {})

    api.startup(fail_fast=False, cli_model_path=tmp_path / "cli-buffalo-l")

    assert captured["name"] == "insightface"
    assert captured["kwargs"] == {"model_path": tmp_path / "cli-buffalo-l", "gui_model_path": None}
    assert api.get_recognizer() == "recognizer"


def test_startup_exits_with_guidance_when_model_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        api,
        "settings",
        SimpleNamespace(
            model_name="insightface",
            threshold=0.3,
            gallery_path=tmp_path / "gallery.pkl",
            registered_dir=tmp_path / "registered",
            identities_csv=tmp_path / "identities.csv",
            max_upload_bytes=1024,
        ),
    )
    monkeypatch.setattr(api, "create_model", lambda name, **kwargs: (_ for _ in ()).throw(ModelPathMissingError("missing")))

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        api.startup(fail_fast=True)

    assert exc.value.code == 1
    assert "--model-path" in caplog.text
    assert "config.toml" in caplog.text


def test_startup_exits_with_guidance_when_model_path_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        api,
        "settings",
        SimpleNamespace(
            model_name="insightface",
            threshold=0.3,
            gallery_path=tmp_path / "gallery.pkl",
            registered_dir=tmp_path / "registered",
            identities_csv=tmp_path / "identities.csv",
            max_upload_bytes=1024,
        ),
    )
    monkeypatch.setattr(api, "create_model", lambda name, **kwargs: (_ for _ in ()).throw(ModelNotFoundError("bad path")))

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        api.startup(fail_fast=True)

    assert exc.value.code == 1
    assert "--model-path" in caplog.text
    assert "bad path" in caplog.text


def test_startup_exits_with_guidance_when_model_load_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        api,
        "settings",
        SimpleNamespace(
            model_name="insightface",
            threshold=0.3,
            gallery_path=tmp_path / "gallery.pkl",
            registered_dir=tmp_path / "registered",
            identities_csv=tmp_path / "identities.csv",
            max_upload_bytes=1024,
        ),
    )
    monkeypatch.setattr(api, "create_model", lambda name, **kwargs: (_ for _ in ()).throw(ModelLoadError("onnxruntime failed")))

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        api.startup(fail_fast=True)

    assert exc.value.code == 1
    assert "损坏或不兼容" in caplog.text
    assert "onnxruntime failed" in caplog.text


def test_recognize_returns_readable_500_and_service_stays_alive_on_model_load_error(monkeypatch) -> None:
    class FlakyRecognizer:
        def __init__(self) -> None:
            self.calls = 0

        def recognize_image(self, image) -> list[dict]:
            self.calls += 1
            if self.calls == 1:
                raise ModelLoadError("runtime inference failed")
            return []

    recognizer = FlakyRecognizer()
    client = TestClient(app)
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buffer, format="PNG")
    payload = buffer.getvalue()

    monkeypatch.setattr(api, "get_recognizer", lambda: recognizer)

    first = client.post("/recognize", files={"file": ("ok.png", payload, "image/png")})
    second = client.post("/recognize", files={"file": ("ok.png", payload, "image/png")})

    assert first.status_code == 500
    assert "runtime inference failed" in first.json()["message"]
    assert second.status_code == 200
    assert second.json() == []
