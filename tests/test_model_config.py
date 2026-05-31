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
