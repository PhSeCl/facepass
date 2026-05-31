from fastapi.testclient import TestClient
from types import SimpleNamespace
from PIL import Image
import io

import src.backend.api as api

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
