from pathlib import Path

import numpy as np
import pytest

import src.face_model.insightface_model as insightface_model_module
from src.face_model.insightface_model import InsightFaceModel


class FakeRecognitionModel:
    def get_feat(self, image: np.ndarray) -> np.ndarray:
        return np.array([[3.0, 4.0]], dtype=np.float32)


class FakeFaceAnalysisApp:
    def __init__(self) -> None:
        self.models = {"recognition": FakeRecognitionModel()}
        self.prepare_calls: list[tuple[int, tuple[int, int]]] = []

    def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
        self.prepare_calls.append((ctx_id, det_size))


def test_encode_aligned_uses_recognition_model_without_detection() -> None:
    model = object.__new__(InsightFaceModel)
    model.recognition_model = FakeRecognitionModel()

    embedding = model.encode_aligned(np.zeros((8, 8, 3), dtype=np.uint8))

    assert np.allclose(embedding, np.array([0.6, 0.8], dtype=np.float32))


def test_insightface_model_resolves_validates_and_persists_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_path = tmp_path / "resolved-buffalo-l"
    app = FakeFaceAnalysisApp()
    calls: dict[str, object] = {}
    persisted: list[Path] = []

    def fake_resolve_model_path(cli_path=None, gui_path=None):
        calls["resolve"] = (cli_path, gui_path)
        return resolved_path

    def fake_validate_model_path(path: Path) -> Path:
        calls["validate"] = path
        return resolved_path

    def fake_create_face_analysis_app(model_path: Path, model_name: str, providers: list[str] | None):
        calls["create"] = (model_path, model_name, providers)
        return app

    monkeypatch.setattr(insightface_model_module.model_config, "resolve_model_path", fake_resolve_model_path)
    monkeypatch.setattr(insightface_model_module.model_config, "validate_model_path", fake_validate_model_path)
    monkeypatch.setattr(insightface_model_module.model_config, "persist_path", persisted.append)
    monkeypatch.setattr(insightface_model_module, "_create_face_analysis_app", fake_create_face_analysis_app)

    model = InsightFaceModel(model_path=tmp_path / "cli-model")

    assert calls["resolve"] == (tmp_path / "cli-model", None)
    assert calls["validate"] == resolved_path
    assert calls["create"] == (resolved_path, "buffalo_l", None)
    assert persisted == [resolved_path]
    assert model.recognition_model is app.models["recognition"]
    assert app.prepare_calls == [(0, (640, 640))]


def test_insightface_model_does_not_persist_when_using_default_config_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_path = tmp_path / "persisted-buffalo-l"
    persisted: list[Path] = []

    monkeypatch.setattr(insightface_model_module.model_config, "resolve_model_path", lambda cli_path=None, gui_path=None: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "validate_model_path", lambda path: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "persist_path", persisted.append)
    monkeypatch.setattr(
        insightface_model_module,
        "_create_face_analysis_app",
        lambda model_path, model_name, providers: FakeFaceAnalysisApp(),
    )

    InsightFaceModel()

    assert persisted == []


def test_insightface_model_does_not_persist_explicit_path_when_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_path = tmp_path / "resolved-buffalo-l"
    persisted: list[Path] = []

    monkeypatch.setattr(insightface_model_module.model_config, "resolve_model_path", lambda cli_path=None, gui_path=None: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "validate_model_path", lambda path: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "persist_path", persisted.append)
    monkeypatch.setattr(
        insightface_model_module,
        "_create_face_analysis_app",
        lambda model_path, model_name, providers: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        InsightFaceModel(model_path=tmp_path / "cli-model")

    assert persisted == []
