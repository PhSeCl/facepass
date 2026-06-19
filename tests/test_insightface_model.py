from pathlib import Path

import numpy as np
import pytest

import src.face_model.insightface_model as insightface_model_module
from src.common.errors import ModelLoadError
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


def test_default_execution_providers_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        insightface_model_module.importlib,
        "import_module",
        lambda name: type(
            "FakeOrtModule",
            (),
            {"get_available_providers": staticmethod(lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])},
        )(),
    )

    providers = insightface_model_module._default_execution_providers()

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_default_execution_providers_falls_back_to_cpu_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        insightface_model_module.importlib,
        "import_module",
        lambda name: type(
            "FakeOrtModule",
            (),
            {"get_available_providers": staticmethod(lambda: ["AzureExecutionProvider", "CPUExecutionProvider"])},
        )(),
    )

    providers = insightface_model_module._default_execution_providers()

    assert providers == ["CPUExecutionProvider"]


def test_runtime_diagnostics_reports_cuda_when_provider_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        insightface_model_module.importlib,
        "import_module",
        lambda name: type(
            "FakeOrtModule",
            (),
            {
                "__version__": "9.9.9",
                "get_available_providers": staticmethod(
                    lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ),
                "get_device": staticmethod(lambda: "GPU"),
            },
        )(),
    )

    diagnostics = insightface_model_module.get_runtime_diagnostics()

    assert diagnostics["onnxruntime_version"] == "9.9.9"
    assert diagnostics["device"] == "GPU"
    assert diagnostics["available_providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert diagnostics["preferred_providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert diagnostics["gpu_enabled"] is True


def test_runtime_diagnostics_reports_cpu_when_cuda_provider_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        insightface_model_module.importlib,
        "import_module",
        lambda name: type(
            "FakeOrtModule",
            (),
            {
                "__version__": "1.2.3",
                "get_available_providers": staticmethod(
                    lambda: ["AzureExecutionProvider", "CPUExecutionProvider"]
                ),
                "get_device": staticmethod(lambda: "CPU"),
            },
        )(),
    )

    diagnostics = insightface_model_module.get_runtime_diagnostics()

    assert diagnostics["onnxruntime_version"] == "1.2.3"
    assert diagnostics["device"] == "CPU"
    assert diagnostics["available_providers"] == ["AzureExecutionProvider", "CPUExecutionProvider"]
    assert diagnostics["preferred_providers"] == ["CPUExecutionProvider"]
    assert diagnostics["gpu_enabled"] is False


def test_maybe_preload_gpu_dlls_loads_cuda_runtime_when_cuda_provider_is_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, bool, bool, object]] = []

    monkeypatch.setattr(
        insightface_model_module.importlib,
        "import_module",
        lambda name: type(
            "FakeOrtModule",
            (),
            {
                "preload_dlls": staticmethod(
                    lambda cuda=True, cudnn=True, msvc=True, directory=None: calls.append(
                        (cuda, cudnn, msvc, directory)
                    )
                )
            },
        )(),
    )

    insightface_model_module._maybe_preload_gpu_dlls(["CUDAExecutionProvider", "CPUExecutionProvider"])

    assert calls == [(True, True, True, "")]


def test_maybe_preload_gpu_dlls_skips_preload_when_cuda_provider_is_not_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def fake_import(name: str):
        imported.append(name)
        raise AssertionError("onnxruntime import should not happen")

    monkeypatch.setattr(insightface_model_module.importlib, "import_module", fake_import)

    insightface_model_module._maybe_preload_gpu_dlls(["CPUExecutionProvider"])

    assert imported == []


def test_register_nvidia_dll_directories_adds_bin_dirs_and_preserves_handles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    onnxruntime_package = site_packages / "onnxruntime"
    cudnn_bin = site_packages / "nvidia" / "cudnn" / "bin"
    cublas_bin = site_packages / "nvidia" / "cublas" / "bin"
    onnxruntime_package.mkdir(parents=True)
    cudnn_bin.mkdir(parents=True)
    cublas_bin.mkdir(parents=True)
    (onnxruntime_package / "__init__.py").write_text("# test\n", encoding="utf-8")

    monkeypatch.setattr(insightface_model_module.os, "name", "nt")
    monkeypatch.setattr(insightface_model_module.os, "environ", {"PATH": r"C:\Windows"})
    added: list[str] = []
    monkeypatch.setattr(
        insightface_model_module.os,
        "add_dll_directory",
        lambda path: added.append(path) or f"handle:{path}",
    )
    monkeypatch.setattr(insightface_model_module, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(insightface_model_module, "_REGISTERED_DLL_DIRECTORIES", set())

    fake_module = type("FakeOrtModule", (), {"__file__": str(onnxruntime_package / "__init__.py")})()

    insightface_model_module._register_nvidia_dll_directories(fake_module)

    expected = {str(cublas_bin), str(cudnn_bin)}
    assert set(added) == expected
    assert set(insightface_model_module._REGISTERED_DLL_DIRECTORIES) == expected
    assert set(insightface_model_module._DLL_DIRECTORY_HANDLES) == {f"handle:{path}" for path in expected}
    path_parts = insightface_model_module.os.environ["PATH"].split(insightface_model_module.os.pathsep)
    assert expected.issubset(set(path_parts))


def test_register_nvidia_dll_directories_skips_when_nvidia_root_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    onnxruntime_package = site_packages / "onnxruntime"
    onnxruntime_package.mkdir(parents=True)
    (onnxruntime_package / "__init__.py").write_text("# test\n", encoding="utf-8")

    monkeypatch.setattr(insightface_model_module.os, "name", "nt")
    monkeypatch.setattr(insightface_model_module, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(insightface_model_module, "_REGISTERED_DLL_DIRECTORIES", set())
    monkeypatch.setattr(
        insightface_model_module.os,
        "add_dll_directory",
        lambda path: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    fake_module = type("FakeOrtModule", (), {"__file__": str(onnxruntime_package / "__init__.py")})()

    insightface_model_module._register_nvidia_dll_directories(fake_module)


def test_insightface_model_resolves_and_validates_explicit_path_without_persisting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Loading the model must not write config.toml as a side effect: persisting a
    # path is an explicit, opt-in decision owned by the interactive launcher
    # (scripts/run_dev.py), not a consequence of merely passing model_path.
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
    monkeypatch.setattr(insightface_model_module, "_default_execution_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(insightface_model_module, "_create_face_analysis_app", fake_create_face_analysis_app)

    model = InsightFaceModel(model_path=tmp_path / "cli-model")

    assert calls["resolve"] == (tmp_path / "cli-model", None)
    assert calls["validate"] == resolved_path
    assert calls["create"] == (resolved_path, "buffalo_l", ["CPUExecutionProvider"])
    assert persisted == []
    assert model.recognition_model is app.models["recognition"]
    assert app.prepare_calls == [(0, (640, 640))]


def test_insightface_model_keeps_explicit_provider_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_path = tmp_path / "resolved-buffalo-l"
    app = FakeFaceAnalysisApp()
    calls: dict[str, object] = {}

    def fake_create_face_analysis_app(model_path: Path, model_name: str, providers: list[str] | None):
        calls["create"] = (model_path, model_name, providers)
        return app

    monkeypatch.setattr(insightface_model_module.model_config, "resolve_model_path", lambda cli_path=None, gui_path=None: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "validate_model_path", lambda path: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "persist_path", lambda path: None)
    monkeypatch.setattr(insightface_model_module, "_default_execution_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(insightface_model_module, "_create_face_analysis_app", fake_create_face_analysis_app)

    InsightFaceModel(model_path=tmp_path / "cli-model", providers=["CPUExecutionProvider"])

    assert calls["create"] == (resolved_path, "buffalo_l", ["CPUExecutionProvider"])


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

    with pytest.raises(ModelLoadError, match="load failed"):
        InsightFaceModel(model_path=tmp_path / "cli-model")

    assert persisted == []


def test_insightface_model_wraps_runtime_load_failure_as_model_load_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_path = tmp_path / "resolved-buffalo-l"

    monkeypatch.setattr(insightface_model_module.model_config, "resolve_model_path", lambda cli_path=None, gui_path=None: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "validate_model_path", lambda path: resolved_path)
    monkeypatch.setattr(
        insightface_model_module,
        "_create_face_analysis_app",
        lambda model_path, model_name, providers: (_ for _ in ()).throw(RuntimeError("onnx init failed")),
    )

    with pytest.raises(ModelLoadError, match="onnx init failed"):
        InsightFaceModel(model_path=tmp_path / "cli-model")


def test_detect_and_encode_wraps_runtime_failure_as_model_load_error() -> None:
    class BrokenApp:
        def get(self, image: np.ndarray):
            raise RuntimeError("inference exploded")

    model = object.__new__(InsightFaceModel)
    model.app = BrokenApp()

    with pytest.raises(ModelLoadError, match="inference exploded"):
        model.detect_and_encode(np.zeros((8, 8, 3), dtype=np.uint8))


def test_encode_aligned_wraps_runtime_failure_as_model_load_error() -> None:
    class BrokenRecognitionModel:
        def get_feat(self, image: np.ndarray) -> np.ndarray:
            raise RuntimeError("embedding exploded")

    model = object.__new__(InsightFaceModel)
    model.recognition_model = BrokenRecognitionModel()

    with pytest.raises(ModelLoadError, match="embedding exploded"):
        model.encode_aligned(np.zeros((8, 8, 3), dtype=np.uint8))


def test_detect_and_encode_retries_with_cpu_after_cuda_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_path = tmp_path / "resolved-buffalo-l"
    create_calls: list[tuple[Path, str, list[str]]] = []

    class GpuBrokenApp:
        def __init__(self) -> None:
            self.models = {"recognition": FakeRecognitionModel()}

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            return None

        def get(self, image: np.ndarray):
            raise RuntimeError("CUDNN_BACKEND_API_FAILED")

    class CpuApp:
        def __init__(self) -> None:
            self.models = {"recognition": FakeRecognitionModel()}

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            return None

        def get(self, image: np.ndarray):
            face = type(
                "Face",
                (),
                {
                    "bbox": np.array([1, 2, 11, 12], dtype=np.float32),
                    "embedding": np.array([3.0, 4.0], dtype=np.float32),
                    "det_score": 0.9,
                    "kps": None,
                },
            )()
            return [face]

    def fake_create_face_analysis_app(model_path: Path, model_name: str, providers: list[str]):
        create_calls.append((model_path, model_name, providers))
        if providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]:
            return GpuBrokenApp()
        if providers == ["CPUExecutionProvider"]:
            return CpuApp()
        raise AssertionError(f"unexpected providers: {providers}")

    monkeypatch.setattr(insightface_model_module.model_config, "resolve_model_path", lambda cli_path=None, gui_path=None: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "validate_model_path", lambda path: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "persist_path", lambda path: None)
    monkeypatch.setattr(insightface_model_module, "_create_face_analysis_app", fake_create_face_analysis_app)

    model = InsightFaceModel(model_path=tmp_path / "cli-model", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    faces = model.detect_and_encode(np.zeros((8, 8, 3), dtype=np.uint8))

    assert create_calls == [
        (resolved_path, "buffalo_l", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        (resolved_path, "buffalo_l", ["CPUExecutionProvider"]),
    ]
    assert len(faces) == 1
    assert faces[0].bbox == (1, 2, 10, 10)


def test_encode_aligned_retries_with_cpu_after_cuda_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_path = tmp_path / "resolved-buffalo-l"
    create_calls: list[tuple[Path, str, list[str]]] = []

    class BrokenRecognitionModel:
        def get_feat(self, image: np.ndarray) -> np.ndarray:
            raise RuntimeError("CUDNN_BACKEND_API_FAILED")

    class WorkingRecognitionModel:
        def get_feat(self, image: np.ndarray) -> np.ndarray:
            return np.array([[3.0, 4.0]], dtype=np.float32)

    class App:
        def __init__(self, recognition_model) -> None:
            self.models = {"recognition": recognition_model}

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            return None

    def fake_create_face_analysis_app(model_path: Path, model_name: str, providers: list[str]):
        create_calls.append((model_path, model_name, providers))
        if providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]:
            return App(BrokenRecognitionModel())
        if providers == ["CPUExecutionProvider"]:
            return App(WorkingRecognitionModel())
        raise AssertionError(f"unexpected providers: {providers}")

    monkeypatch.setattr(insightface_model_module.model_config, "resolve_model_path", lambda cli_path=None, gui_path=None: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "validate_model_path", lambda path: resolved_path)
    monkeypatch.setattr(insightface_model_module.model_config, "persist_path", lambda path: None)
    monkeypatch.setattr(insightface_model_module, "_create_face_analysis_app", fake_create_face_analysis_app)

    model = InsightFaceModel(model_path=tmp_path / "cli-model", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    embedding = model.encode_aligned(np.zeros((8, 8, 3), dtype=np.uint8))

    assert create_calls == [
        (resolved_path, "buffalo_l", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        (resolved_path, "buffalo_l", ["CPUExecutionProvider"]),
    ]
    assert np.allclose(embedding, np.array([0.6, 0.8], dtype=np.float32))
