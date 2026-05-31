import logging
from pathlib import Path

import pytest

from src.common.errors import (
    FacePassError,
    ModelConfigError,
    ModelIncompleteError,
    ModelLoadError,
    ModelNotFoundError,
    ModelPathMissingError,
)
from src.face_model import model_config


BUFFALO_L_REQUIRED_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)


def _create_buffalo_l_dir(base_dir: Path, *, empty_files: set[str] | None = None, missing_files: set[str] | None = None) -> Path:
    model_dir = base_dir / "buffalo_l"
    model_dir.mkdir()
    empty_files = empty_files or set()
    missing_files = missing_files or set()

    for filename in BUFFALO_L_REQUIRED_FILES:
        if filename in missing_files:
            continue
        contents = b"" if filename in empty_files else b"onnx"
        (model_dir / filename).write_bytes(contents)

    return model_dir


def test_model_config_errors_extend_expected_base_classes() -> None:
    assert issubclass(ModelConfigError, FacePassError)
    assert issubclass(ModelPathMissingError, ModelConfigError)
    assert issubclass(ModelNotFoundError, ModelConfigError)
    assert issubclass(ModelIncompleteError, ModelConfigError)
    assert issubclass(ModelLoadError, ModelConfigError)


def test_load_persisted_path_returns_none_when_config_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_config, "CONFIG_FILE", tmp_path / "config.toml")

    assert model_config.load_persisted_path() is None


def test_persist_path_round_trips_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.toml"
    model_path = tmp_path / "models" / "buffalo_l"
    monkeypatch.setattr(model_config, "CONFIG_FILE", config_file)

    model_config.persist_path(model_path)

    assert model_config.load_persisted_path() == model_path.resolve()


def test_load_persisted_path_returns_none_and_warns_for_invalid_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[model\npath = 'broken'\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "CONFIG_FILE", config_file)

    with caplog.at_level(logging.WARNING):
        loaded_path = model_config.load_persisted_path()

    assert loaded_path is None
    assert "config.toml" in caplog.text


def test_persist_path_updates_model_path_without_losing_other_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[app]\nname = \"facepass\"\n\n[model]\npath = \"/old/model\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "CONFIG_FILE", config_file)

    model_config.persist_path(tmp_path / "new-model")

    contents = config_file.read_text(encoding="utf-8")
    assert 'name = "facepass"' in contents
    assert model_config.load_persisted_path() == (tmp_path / "new-model").resolve()


def test_resolve_model_path_prefers_cli_over_gui_and_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[model]\npath = "/persisted/model"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "CONFIG_FILE", config_file)

    resolved = model_config.resolve_model_path(
        cli_path=tmp_path / "cli-model",
        gui_path=tmp_path / "gui-model",
    )

    assert resolved == (tmp_path / "cli-model").resolve()


def test_resolve_model_path_uses_gui_when_cli_missing(tmp_path: Path) -> None:
    resolved = model_config.resolve_model_path(
        cli_path=None,
        gui_path=tmp_path / "gui-model",
    )

    assert resolved == (tmp_path / "gui-model").resolve()


def test_resolve_model_path_uses_persisted_path_when_no_explicit_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.toml"
    persisted = tmp_path / "persisted-model"
    config_file.write_text(f'[model]\npath = "{persisted.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "CONFIG_FILE", config_file)

    resolved = model_config.resolve_model_path()

    assert resolved == persisted.resolve()


def test_resolve_model_path_raises_when_no_source_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_config, "CONFIG_FILE", tmp_path / "config.toml")

    with pytest.raises(ModelPathMissingError):
        model_config.resolve_model_path()


def test_resolve_model_path_resolves_relative_paths_from_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_config, "_project_root", lambda: tmp_path)

    resolved = model_config.resolve_model_path(cli_path=Path("models/buffalo_l"))

    assert resolved == (tmp_path / "models" / "buffalo_l").resolve()


def test_validate_model_path_raises_when_path_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        model_config.validate_model_path(tmp_path / "missing-model")


def test_validate_model_path_raises_when_required_file_is_missing(tmp_path: Path) -> None:
    model_dir = _create_buffalo_l_dir(tmp_path, missing_files={"w600k_r50.onnx"})

    with pytest.raises(ModelIncompleteError, match="w600k_r50.onnx"):
        model_config.validate_model_path(model_dir)


def test_validate_model_path_raises_when_required_file_is_empty(tmp_path: Path) -> None:
    model_dir = _create_buffalo_l_dir(tmp_path, empty_files={"det_10g.onnx"})

    with pytest.raises(ModelIncompleteError, match="det_10g.onnx"):
        model_config.validate_model_path(model_dir)


def test_validate_model_path_returns_resolved_path_for_complete_directory(tmp_path: Path) -> None:
    model_dir = _create_buffalo_l_dir(tmp_path)

    assert model_config.validate_model_path(model_dir) == model_dir.resolve()
