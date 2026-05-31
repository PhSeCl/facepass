import asyncio
import io
from types import SimpleNamespace

import httpx
from PIL import Image

import src.backend.api as api


def _write_image(path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 8), color=color).save(path)


def test_fake_model_runs_gallery_build_and_recognition_via_http(monkeypatch, tmp_path) -> None:
    registered_dir = tmp_path / "registered" / "p01"
    registered_dir.mkdir(parents=True)
    _write_image(registered_dir / "known.png", (255, 0, 0))
    identities_csv = tmp_path / "identities.csv"
    identities_csv.write_text("identity_id,name\np01,Alice\n", encoding="utf-8")

    settings = SimpleNamespace(
        model_name="fake",
        threshold=0.5,
        gallery_path=tmp_path / "gallery.pkl",
        registered_dir=registered_dir.parent,
        identities_csv=identities_csv,
        max_upload_bytes=1024 * 1024,
    )
    monkeypatch.setattr(api, "settings", settings)
    monkeypatch.setattr(api, "_recognizer", None)
    monkeypatch.setattr(api, "_gallery", api.Gallery())
    monkeypatch.setattr(api, "_id2name", {})

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buffer, format="PNG")

    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=api.app)
        async with api.app.router.lifespan_context(api.app):
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    "/recognize",
                    files={"file": ("known.png", buffer.getvalue(), "image/png")},
                )

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == [
        {
            "bbox": [0, 0, 8, 8],
            "identity_id": "p01",
            "name": "Alice",
            "similarity": 1.0,
            "is_unknown": False,
        }
    ]
