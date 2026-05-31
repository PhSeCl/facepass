import importlib

import src.backend.config as backend_config
import src.face_model.model_config as model_config


def test_settings_threshold_reads_value_from_config_toml(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[recognition]\nthreshold = 0.42\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "CONFIG_FILE", config_file)

    settings = backend_config.Settings()

    assert settings.threshold == 0.42


def test_module_level_settings_threshold_falls_back_to_default_when_config_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_config, "CONFIG_FILE", tmp_path / "config.toml")

    reloaded = importlib.reload(backend_config)

    assert reloaded.settings.threshold == 0.30
