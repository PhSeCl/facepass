import logging
from fastapi.testclient import TestClient
from types import SimpleNamespace
from PIL import Image
import io
import base64

import pytest

import src.backend.api as api
from src.common.errors import ModelLoadError, ModelNotFoundError, ModelPathMissingError
from src.backend.dataset_import import DatasetArchiveError, DatasetLayoutError

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


def test_dataset_inspect_reports_registered_presence(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(api, "inspect_external_dataset_archive", lambda path: True)

    response = client.post(
        "/dataset-eval/inspect",
        files={"file": ("test.zip", b"zip-bytes", "application/zip")},
    )

    assert response.status_code == 200
    assert response.json() == {"has_registered": True}


def test_dataset_inspect_returns_readable_archive_error(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        api,
        "inspect_external_dataset_archive",
        lambda path: (_ for _ in ()).throw(DatasetArchiveError("压缩包无法解压,可能已损坏")),
    )

    response = client.post(
        "/dataset-eval/inspect",
        files={"file": ("bad.zip", b"bad", "application/zip")},
    )

    assert response.status_code == 400
    assert "压缩包无法解压" in response.json()["message"]


def test_dataset_eval_returns_structured_report(monkeypatch) -> None:
    client = TestClient(app)
    transparent_png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode("ascii")
    captured: dict[str, object] = {}

    def fake_run_external_eval(*args, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            gallery_source="archive",
            report=SimpleNamespace(
                metrics=SimpleNamespace(
                    strict_top1_accuracy=1.0,
                    matched_top1_accuracy=1.0,
                    detection_recall=1.0,
                    detection_precision=1.0,
                    unknown_detected_accuracy=1.0,
                    predicted_unknown_precision=1.0,
                ),
            ),
            confusion_pairs=[("p01", "p01")],
            missed_detections=[SimpleNamespace(image_name="miss.jpg", bbox=(1, 2, 3, 4))],
            false_positives=[SimpleNamespace(image_name="fp.jpg", bbox=(5, 6, 7, 8))],
            dataset=SimpleNamespace(),
        )

    monkeypatch.setattr(api, "run_external_eval", fake_run_external_eval)
    monkeypatch.setattr(
        api,
        "_render_external_eval_plots",
        lambda dataset, report: {
            "confusion_matrix": f"data:image/png;base64,{transparent_png}",
            "detection_metrics": f"data:image/png;base64,{transparent_png}",
            "accuracy_metrics": f"data:image/png;base64,{transparent_png}",
        },
    )
    monkeypatch.setattr(api, "get_recognizer", lambda: SimpleNamespace(model=object()))
    monkeypatch.setattr(api, "_gallery", api.Gallery())

    response = client.post(
        "/dataset-eval/run",
        data={"gallery_choice": "archive"},
        files={"file": ("test.zip", b"zip-bytes", "application/zip")},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["gallery_source"] == "archive"
    assert payload["metrics"]["strict_top1_accuracy"] == 1.0
    assert payload["plots"]["confusion_matrix"].startswith("data:image/png;base64,")
    assert payload["missed_detections"] == [{"image_name": "miss.jpg", "bbox": [1, 2, 3, 4]}]
    assert payload["false_positives"] == [{"image_name": "fp.jpg", "bbox": [5, 6, 7, 8]}]
    assert captured["kwargs"]["local_gallery"] is api._gallery


def test_dataset_eval_returns_readable_layout_error(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(api, "get_recognizer", lambda: SimpleNamespace(model=object()))
    monkeypatch.setattr(
        api,
        "run_external_eval",
        lambda *args, **kwargs: (_ for _ in ()).throw(DatasetLayoutError("解压成功但未找到 images/ 或标注文件")),
    )

    response = client.post(
        "/dataset-eval/run",
        data={"gallery_choice": "local"},
        files={"file": ("test.zip", b"zip-bytes", "application/zip")},
    )

    assert response.status_code == 400
    assert "未找到 images/" in response.json()["message"]
